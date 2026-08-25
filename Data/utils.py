from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
import re
import string
import unicodedata

# NLP Dataset Cleaning
#--------------------

def clean_corpus(
    corpus,
    lower=True,
    remove_punct=True,
    remove_digits=True,
):
    
    cleaned_corpus = []
    for doc in corpus:
        doc = unicodedata.normalize('NFKD', doc)
        doc = doc.encode('ascii', 'ignore').decode('utf-8', 'ignore')
        
        if lower:
            doc = doc.lower()
        if remove_punct:
            doc = doc.translate(str.maketrans('', '', string.punctuation))
        if remove_digits:
            doc = re.sub(r'\d+', '', doc)
        
        doc = re.sub(r'\s+', ' ', doc).strip()
        
        tokens = doc.split()
        
        cleaned_text = " ".join(tokens)
        
        # delete empty docs
        if cleaned_text.strip():
            cleaned_corpus.append(cleaned_text)
    
    return cleaned_corpus


def unify_text_column(dataset, dataset_name):

    if dataset_name == "sst2":
        return dataset.rename_column("sentence", "text")

    if dataset_name == "sms":
        return dataset.rename_column("sms", "text")

    return dataset

def preprocess(dataset, column='text'):

    def clean_text(example):
        text = example[column]

        if text is None:
            text = ""

        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\d+", " ", text)
        text = re.sub(r"\W+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        example[column] = text
        return example

    # Appliquer le nettoyage
    dataset = dataset.map(clean_text)

    # Supprimer les textes vides
    dataset = dataset.filter(lambda x: x[column] != "")

    return dataset



# Dataset Importing
#--------------------

def import_dataset(name="20newsgroups", full_dataset_=False, batch_size=64):

    print(f"{name} dataset importing .... \n\n")

    # *****************************
    if name == "20newsgroups":
        dataset = load_dataset("SetFit/20_newsgroups")

        # Nettoyage des textes
        dataset = dataset.map(lambda x: {"text": clean_corpus([x["text"]])[0] if clean_corpus([x["text"]]) else ""})
        dataset = dataset.filter(lambda x: len(x["text"]) > 0)

        train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=True)

        if full_dataset_:
            full_dataset = concatenate_datasets([dataset['train'], dataset['test']])
            return DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader
  
  # *****************************
    if name == "reuters":

        dataset = load_dataset('ucirvine/reuters21578', 'ModApte', trust_remote_code=True)  #ModHayes  ModLewis

        train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=True)
        
        if full_dataset_:
            full_dataset = concatenate_datasets([dataset['train'], dataset['test']])
            return DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader

  # *****************************
    if name == "wos":

        dataset = load_dataset("HDLTex/web_of_science", 'WOS46985') 

        return DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)

  # *****************************
    if name == "dbpedia14":

        dataset = load_dataset("fancyzhx/dbpedia_14")
        
        train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=True)
        
        if full_dataset_:
            full_dataset = concatenate_datasets([dataset['train'], dataset['test']])
            return DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader

    # ***************************
    if name == "agnews": 
        
        dataset = load_dataset("fancyzhx/ag_news")

        train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=True)
        
        if full_dataset_:
            full_dataset = concatenate_datasets([dataset['train'], dataset['test']])
            return DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader


    # ***************************
    if name == "sms": 
        
        dataset = load_dataset("ucirvine/sms_spam")

        dataset_split = dataset['train'].train_test_split(
            test_size=0.2,              
            stratify_by_column="label", 
            seed=42                     
        )

        train_dataloader = DataLoader(dataset_split["train"], batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset_split["test"], batch_size=batch_size, shuffle=True)
        
        if full_dataset_:
            full_dataset = concatenate_datasets([dataset['train'], dataset['test']])
            return DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader

    # ***************************
    if name == "enron": 
        
        dataset = load_dataset("SetFit/enron_spam")

        train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=True)
        
        if full_dataset_:
            full_dataset = concatenate_datasets([dataset['train'], dataset['test']])
            return DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader
    
    # ***************************
    if name == "imdb": 
        
        dataset = load_dataset("stanfordnlp/imdb")

        train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=True)
        
        if full_dataset_:
            full_dataset = concatenate_datasets([dataset['train'], dataset['test']])
            return DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader
    

    # ***************************
    if name == "sst2": 
        
        dataset = load_dataset("stanfordnlp/sst2")

        train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
        # because the testset is not labeled, the validation one is
        test_dataloader = DataLoader(dataset['validation'], batch_size=batch_size, shuffle=True)
        
        if full_dataset_:
            full_dataset = concatenate_datasets([dataset['train'], dataset['validation']])
            return DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader
    
    # ***************************
    if name == "mage": 
        
        dataset = load_dataset("yaful/MAGE")

        train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=True)
        
        if full_dataset_:
            full_dataset = concatenate_datasets([dataset['train'], dataset['test']])
            return DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader
    

    # ***************************
    if name == "m4": 

        def load_jsonl(filepath):
            data = []
            with open(filepath, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"[WARNING] skip ligne {i} dans {filepath}")
            return data


        def parse_filename(filename):
            name = filename.replace(".jsonl", "")
            parts = name.split("_")

            domain = parts[0].lower()
            generator = parts[1].lower()

            return domain, generator

        def convert_entry(entry, file_domain, generator):
            samples = []
            domain = file_domain

            # human
            if entry.get("human_text"):
                samples.append({
                    "text": entry["human_text"],
                    "label": 1,
                    "generator": "human",
                    "inlier_topic": domain,   
                    "prompt": entry.get("prompt", ""),
                    "source_id": entry.get("source_ID", "")
                })

            # machine
            if entry.get("machine_text"):
                samples.append({
                    "text": entry["machine_text"],
                    "label": 0,
                    "generator": generator,
                    "inlier_topic": domain,  
                    "prompt": entry.get("prompt", ""),
                    "source_id": entry.get("source_ID", "")
                })

            return samples

        features = Features({
            "text": Value("string"),
            "label": Value("int64"),
            "generator": Value("string"),
            "inlier_topic": Value("string"),
            "prompt": Value("string"),
            "source_id": Value("string"),
        })

        def load_m4_dataset(data_dir="Anomaly Detection Framework/m4_data"):
            all_samples = []

            for file in os.listdir(data_dir):
                if not file.endswith(".jsonl"):
                    continue

                filepath = os.path.join(data_dir, file)

                file_domain, generator = parse_filename(file)
                raw_data = load_jsonl(filepath)

                for entry in raw_data:
                    all_samples.extend(
                        convert_entry(entry, file_domain, generator)
                    )

            dataset = Dataset.from_list(all_samples, features=features)

            return dataset


        def add_stratify_key(example):
            example["stratify_key"] = f"{example['inlier_topic']}_{example['label']}"
            return example


        def create_dataloaders(dataset, batch_size=32, test_size=0.2, seed=42):

            dataset = dataset.map(add_stratify_key)

            unique_keys = list(set(dataset["stratify_key"]))
            dataset = dataset.cast_column(
                "stratify_key",
                ClassLabel(names=unique_keys)
            )

            split = dataset.train_test_split(
                test_size=test_size,
                stratify_by_column="stratify_key",
                seed=seed
            )

            train_loader = DataLoader(
                split["train"],
                batch_size=batch_size,
                shuffle=True
            )

            test_loader = DataLoader(
                split["test"],
                batch_size=batch_size,
                shuffle=False
            )

            return train_loader, test_loader


        data = load_m4_dataset('/home/2017025/ayouce01/Textual-Anomaly-Detection-Framework/Anomaly Detection Framework/m4_data')

        return create_dataloaders(
            data,
            batch_size=16
            )


    raise Exception("The dataset naming doesn't correspond !")