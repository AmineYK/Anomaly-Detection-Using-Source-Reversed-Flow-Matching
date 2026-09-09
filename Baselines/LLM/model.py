import json
import random
import re

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from Modelisation.Baselines.baseline import BaselineModel


_INLIER_EXAMPLES_BLOCK = """
## Normal Text Examples (from training set):
The following are real examples of **normal** (non-anomalous) texts from the training set.
Use them as concrete reference to calibrate what a normal text looks like.
{inlier_block}
"""

_ANOMALY_EXAMPLES_BLOCK = """
## Anomaly Text Examples:
The following are real examples of **anomalous** texts that do NOT belong to the normal category.
Use them as reference to understand what an anomaly looks like.
{anomaly_block}
"""

_EXAMPLE_LINE = '  - Example {idx}: "{text}"'

_PROMPT_TEMPLATE = """You are an intelligent and professional assistant that detects anomalies in text data.

## Task:
Following the rules below, determine whether the given text sample is an anomaly. Provide a brief explanation of your reasoning and assign an anomaly confidence score between 0 and 1.

## Normal Category:
{categories}
{inlier_section}{anomaly_section}
## Rules:
1. **Anomaly Definition**:
   - A text sample is considered an **anomaly** if it does **not** belong to the normal category listed above.

2. **Scoring**:
   - Assign an anomaly confidence score between 0 and 1.
   - Use higher scores (close to 1) when you are highly confident the text IS an anomaly.
   - Use lower scores (close to 0) when the text belongs to the normal category.

3. **Step-by-step Reasoning**:
   - Step 1. Read the entire text sample carefully.
   - Step 2. Compare it to the normal category{examples_hint}.
   - Step 3. Determine if it aligns with the normal category.
     - If yes -> not an anomaly (score close to 0).
     - If no  -> anomaly (score close to 1).
   - Step 4. Assign the anomaly confidence score.

4. **Additional Notes**:
   - If uncertain, assume it is NOT an anomaly.

5. **Response Format**:
   - Respond ONLY in valid JSON with exactly two keys: "reason" and "anomaly_score".
   - "reason": one to three sentences of reasoning.
   - "anomaly_score": a float between 0 and 1.
   - Example: {{"reason": "The text is about sports.", "anomaly_score": 0.05}}

Text sample: "{text}"

Response in JSON format:"""


class LLMAnomalyDetector(BaselineModel):

    MODEL_OPTIONS = {
        "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3"
    }

    def __init__(
        self,
        normal_category,
        model_name="Qwen/Qwen2.5-3B-Instruct",
        n_inlier_examples=3,
        outlier_examples=None,
        threshold=0.5,
        max_new_tokens=256,
        temperature=0.0,
        seed=42,
        verbose=True,
    ):
        super().__init__(
            normal_category,
            model_name,
            n_inlier_examples,
            outlier_examples,
            threshold,
            max_new_tokens,
            temperature,
            seed,
        )

        self.normal_category = normal_category
        self.model_name = model_name
        self.n_inlier_examples = n_inlier_examples
        self.outlier_examples = outlier_examples or []
        self.threshold = threshold
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.seed = seed
        self.verbose = verbose

        self._train_texts = []
        self._inlier_sample = []

        self._tokenizer = None
        self._model = None
        self._device = None

    def _load_model(self):
        """Load the tokenizer and model once."""
        if self._model is not None:
            return

        print(f"[INFO] Loading model: {self.model_name}")

        self._device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"[INFO] Device: {self._device}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=(
                torch.float16
                if self._device == "cuda"
                else torch.float32
            ),
            device_map="auto",
            trust_remote_code=True,
        )

        self._model.eval()

        print("[INFO] Model loaded.\n")

    def train(self, X_train):
        """
        Store the training texts and select inlier examples
        for the inference prompt.
        """
        if not X_train:
            raise ValueError("X_train must not be empty.")

        self._train_texts = list(X_train)

        random.seed(self.seed)

        n = min(
            self.n_inlier_examples,
            len(self._train_texts),
        )

        self._inlier_sample = (
            random.sample(self._train_texts, n)
            if n > 0
            else []
        )

        mode = (
            f"{n} inlier(s)"
            if n > 0
            else "zero-shot"
        )

        outliers = (
            f" + {len(self.outlier_examples)} outlier(s)"
            if self.outlier_examples
            else ""
        )

        print(
            f"[train] {len(X_train)} texts stored | "
            f"prompt mode: {mode}{outliers}"
        )

        return self

    def test(self, X_test, y_test=None):
        """
        Detect anomalies in X_test.

        If y_test is provided, evaluation metrics are computed.
        """
        if not self._train_texts:
            raise RuntimeError("Call train() before test().")

        self._load_model()

        results = []

        for i, text in enumerate(tqdm(X_test)):
            if self.verbose:
                preview = (
                    text[:65] + "..."
                    if len(text) > 65
                    else text
                )

                print(
                    f"[{i + 1}/{len(X_test)}] '{preview}'"
                )

            result = self._predict_one(text)
            results.append(result)

            if self.verbose:
                label = (
                    "ANOMALY"
                    if result["is_anomaly"]
                    else "NORMAL "
                )

                print(
                    f"  -> [{label}] "
                    f"score={result['anomaly_score']:.3f} "
                    f"| inliers={result['n_inliers']} "
                    f"outliers={result['n_outliers']}"
                )

                print(f"  -> {result['reason']}\n")

        if y_test is not None:
            return self._evaluate(results, y_test)

        return results

    def _predict_one(self, text):
        """Run inference on a single text."""
        prompt = self._build_prompt(text)

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        try:
            input_ids = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self._device)
        except Exception:
            input_ids = self._tokenizer(
                prompt,
                return_tensors="pt",
            ).input_ids.to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[-1]:]

        raw_output = self._tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        parsed = self._parse_response(raw_output)

        anomaly_score = float(
            parsed.get("anomaly_score", 0.5)
        )

        return {
            "text": text,
            "is_anomaly": anomaly_score >= self.threshold,
            "anomaly_score": anomaly_score,
            "reason": parsed.get("reason", ""),
            "raw_output": raw_output,
            "n_inliers": len(self._inlier_sample),
            "n_outliers": len(self.outlier_examples),
        }

    def _build_prompt(self, text):
        """Build the prompt using the normal and anomaly examples."""
        categories_str = f"- **{self.normal_category}**"

        if self._inlier_sample:
            lines = "\n".join(
                _EXAMPLE_LINE.format(
                    idx=i + 1,
                    text=sample.replace('"', '\\"'),
                )
                for i, sample in enumerate(
                    self._inlier_sample
                )
            )

            inlier_section = (
                _INLIER_EXAMPLES_BLOCK.format(
                    inlier_block=lines
                )
            )
        else:
            inlier_section = ""

        if self.outlier_examples:
            lines = "\n".join(
                _EXAMPLE_LINE.format(
                    idx=i + 1,
                    text=sample.replace('"', '\\"'),
                )
                for i, sample in enumerate(
                    self.outlier_examples
                )
            )

            anomaly_section = (
                _ANOMALY_EXAMPLES_BLOCK.format(
                    anomaly_block=lines
                )
            )
        else:
            anomaly_section = ""

        hints = []

        if self._inlier_sample:
            hints.append("the normal examples")

        if self.outlier_examples:
            hints.append("the anomaly examples")

        examples_hint = (
            ", and compare it against "
            + " and ".join(hints)
            if hints
            else ""
        )

        return _PROMPT_TEMPLATE.format(
            categories=categories_str,
            inlier_section=inlier_section,
            anomaly_section=anomaly_section,
            examples_hint=examples_hint,
            text=text.replace('"', '\\"'),
        )

    @staticmethod
    def _parse_response(response_text):
        """Extract the JSON response generated by the LLM."""
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass

        match = re.search(
            r"\{.*?\}",
            response_text,
            re.DOTALL,
        )

        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        score_match = re.search(
            r'"anomaly_score"\s*:\s*([0-9.]+)',
            response_text,
        )

        reason_match = re.search(
            r'"reason"\s*:\s*"([^"]+)"',
            response_text,
        )

        if score_match:
            return {
                "reason": (
                    reason_match.group(1)
                    if reason_match
                    else "Could not parse reason."
                ),
                "anomaly_score": float(
                    score_match.group(1)
                ),
            }

        return {
            "reason": "Failed to parse model output.",
            "anomaly_score": 0.5,
        }

    @staticmethod
    def _evaluate(results, y_test):
        """Compute AUC-ROC, FPR@95, and Average Precision."""

        def fpr95_score(y_true, scores):
            fpr, tpr, _ = roc_curve(
                y_true,
                scores,
                pos_label=1,
            )

            idx = np.where(tpr >= 0.95)[0][0]

            return fpr[idx]

        scores = [
            result["anomaly_score"]
            for result in results
        ]

        preds = [
            int(result["is_anomaly"])
            for result in results
        ]

        auc = roc_auc_score(y_test, scores)
        fpr95 = fpr95_score(y_test, scores)
        ap = average_precision_score(y_test, scores)

        return auc, fpr95, ap

    def set_outlier_examples(self, examples):
        """Update the anomaly examples used in the prompt."""
        self.outlier_examples = examples

    def __repr__(self):
        return (
            "LLMAnomalyDetector("
            f"model='{self.model_name}', "
            f"n_inlier_examples={self.n_inlier_examples}, "
            f"n_outliers={len(self.outlier_examples)}, "
            f"threshold={self.threshold})"
        )