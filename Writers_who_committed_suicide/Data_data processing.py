##############################
# AUTHORS WHO COMMİTED SUİCİDE
##############################
import textwrap

# imports:
import pandas as pd
pd.set_option("display.expand_frame_repr", False)
import re
import random
import textwrap

####################
# DATA PREPROCESSİNG
####################
df_book_detail = pd.read_csv('/Users/emrahyilmaz/Miull/Writers _who _committed _suicide/Author_books_detail.csv')
df_authour_detail_copy = pd.read_csv("collect_data/web_scraping/author_name.csv")
df_author = df_authour_detail_copy.copy()
df_book = df_book_detail.copy()

# rename columns:
df_book.columns = [col.upper() for col in df_book.columns]
df_author.columns = [col.upper() for col in df_author.columns]

# Merge book and author:
df = pd.merge(df_author, df_book, how='inner', on='NAME')


def general_table(dataframe, head=5):
    print("######### SHAPE #########")
    print(dataframe.shape)
    print("------------------------------")
    print("######### İNFO #########")
    print(dataframe.info())
    print("------------------------------")
    print("######### DTYPES #########")
    print(dataframe.dtypes)
    print("------------------------------")
    print("######### HEAD ############")
    print(dataframe.head(head))
    print("------------------------------")
    print("######### TAİL ############")
    print(dataframe.tail(head))
    print("------------------------------")
    print("########## NA #############")
    print(dataframe.isnull().sum())
    print("------------------------------")
    print("########## QUANTİLES #############")
    print(dataframe.describe().T)
    print("------------------------------")


general_table(df)


df.columns = [col.upper() for col in df.columns]

# create Target:
keys = ["intihar", "kendi hayatına son verdi", "kendisini öldürdü", "hayatına son verdi", "intihar etti",
        "canınına kıydı", "kendini aşağı attı"]


def check_suicide(death, about):
    if pd.isna(death):  # Nan or not nan
        return None
    if isinstance(about, str): # if about is str:
        for key in keys:
            if key in about.lower():
                return 1
    return 0

# Create Target
df['SUİCEDE'] = df.apply(lambda x: check_suicide(x['DEATH'], x['ABOUT_AUTHOR']), axis=1)

df.groupby('SUİCEDE').agg({'SUİCEDE': 'count'})

df.loc[df['NAME'] == 'Jack London', 'SUİCEDE'] = 1
df.loc[df['NAME'] == 'Ernest Hemingway', 'SUİCEDE'] = 1
df.loc[df['NAME'] == 'Sadık Hidayet', 'SUİCEDE'] = 1
df.loc[df['SUİCEDE'] == 1]

# Add new authors who committed suicide:
boosting_authors_detail = pd.read_csv('Boosting_authors_detail.csv')
boosting_book_detail = pd.read_csv('Boosting_author_books_detail.csv')

boosting_book_detail.columns = [col.upper() for col in boosting_book_detail.columns]
boosting_authors_detail.columns = [col.upper() for col in boosting_authors_detail.columns]

df_boosting = pd.merge(boosting_authors_detail, boosting_book_detail, how="inner", on="NAME", )

# Al off the boosting writers kill themself:
df_boosting['SUİCEDE'] = 1

# Concat boosting and normal df:
df = pd.concat([df, df_boosting], ignore_index=True)

# Convert read counts to numeric variables:


def convert_to_numeric(value):
    if isinstance(value, str):
        value = value.replace('/10', '')
        value = value.replace(',', '')
        value = value.replace('.', '')
        value = value.replace('milyon', '00000')
        value = value.replace('bin', '000')

    return float(value)


df['READ'] = df['READ'].apply(convert_to_numeric)
df['RATE'] = df['RATE'].apply(convert_to_numeric)
df['SEEN'] = df['SEEN'].apply(convert_to_numeric)
df['LIKED'] = df['LIKED'].apply(convert_to_numeric)



df.loc[df['BORN'].isnull()]


def extract_year(born):
    if isinstance(born, str):  # Doğru türde olduğundan emin olma
        # Yıl formatı: Dört basamaklı sayı (örneğin: 1881)
        match = re.search(r'\b\d{4}\b', born)
        if match:
            return int(match.group())
    return None


# creat 'BORN_YEAR' as numeric
df['BORN_YEAR'] = df['BORN'].apply(extract_year)

# this column not important and don't have any information
df.drop('LINK', axis=1, inplace=True)

df.loc[df['NAME'] == 'George Orwell', 'BORN_YEAR'] = 1903


def check_author_suicide(dataframe, arthor_name):
    return dataframe.loc[dataframe['NAME'] == arthor_name]




check_author_suicide(dataframe=df, arthor_name='Beşir Fuad')


def extract_country(location):
    if pd.isnull(location):
        return None

    # Yaygın ülke isimlerine göre belirli kurallar ekleyin
    countries = ["ABD", "Osmanlı İmparatorluğu", "Almanya", "Fransa", "İngiltere", "Türkiye", "Japonya", "Rusya",
                 "Rusya İmparatorluğu", "Birleşik Krallık"]

    for country in countries:
        if country in location:
            return country

    return None


df['BORN_COUNTRY'] = df['BORN'].apply(extract_country)

df.head()


#################
# TEXT PROCESSING
#################
# Sentiment analysis complated with kaggle and Bert turkish savasy model

# cleaning df text columns:


def cleaninf_text_df(text):
      text = str(text)
      text = text.lower()
      # Özel karakterleri ve noktalama işaretlerini kaldırma
      text = re.sub(r'[^\w\s]', '', text)
      # Gereksiz boşlukları kaldırma
      text = re.sub(r'\s+', ' ', text).strip()
      return text


df['NAME'] = df['NAME'].apply(lambda x: cleaninf_text_df(x))

df_sentiment = pd.read_csv('collect_data/text_and_sentiment.csv')

df_columns = ['NAME', 'RATE', 'READ', 'LINKS', 'LIKED', 'SEEN', 'BORN', 'DEATH', 'SUİCEDE', 'BORN_YEAR', 'BORN_COUNTRY']

sentiment = ['NAME',  'ABOUT_AUTHOR', 'COMMENT',  'BOOK_1', 'BOOK_2', 'BOOK_3',
             'ABOUT_SENTIMENT', 'BOOK1_SENTIMENT', 'BOOK2_SENTIMENT', 'BOOK3_SENTIMENT', 'COMMENT_sentiment_average']

df = pd.merge(df[df_columns], df_sentiment[sentiment], how='inner', on='NAME')

# Label about author:
df['LABEL_ABOUT'] = None


def labelling(dataframe):
    while True:
        random_index = random.randint(0, len(dataframe) - 1)
        about_author_textwrap = textwrap.fill(dataframe.at[random_index, 'ABOUT_AUTHOR'])
        print(f'YAZARIN ADI: {dataframe.at[random_index, "NAME"]}')
        print(f'YAZAR HAKKINDA: {about_author_textwrap}')

        action = input('(Atla için atla, Programı kapatmak için pas) \n YAZAR ETİKETLERNİ VİRGÜLLERE AYIRARAK GİRİNİZ: ')

        if action == "atla":
            print(f"{random_index} id ES GEÇİLDİ.")
            continue
        elif action == 'pas':
            print(f"ESEN KALIN..... YENGEYE ELİF DEDİN USTA.")
            break

        else:
            label = action.split(',') # virgüllerden böl
            label = [l.strip() for l in label]
            dataframe.at[random_index, 'LABEL_ABOUT'] = label


labelling(df)

df['LABEL_ABOUT'].value_counts()
df['LABEL_ABOUT'].notnull().sum()





df = pd.read_csv("/Users/emrahyilmaz/Desktop/my_model_label.csv", delimiter=";", header=1)
df = df.dropna(subset=['ABOUT_AUTHOR'])

df['ABOUT_AUTHOR'] = df['ABOUT_AUTHOR'].apply(lambda x: cleaninf_text_df(x))
df = df[~df['LABEL_ABOUT'].str.contains("loc", na=False)]

###############
# ABOUT AUTHORS
#############

from sklearn.preprocessing import MultiLabelBinarizer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

model_name = "microsoft/deberta-v3-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=19,
                                                           problem_type="multi_label_classification")


mlb = MultiLabelBinarizer()

y_train = df.loc[df['LABEL_ABOUT'].notnull(),  "LABEL_ABOUT"]
X_train = df.loc[df['LABEL_ABOUT'].notnull(), "ABOUT_AUTHOR"]

y_test = df.loc[df['LABEL_ABOUT'].isnull(), "LABEL_ABOUT"]
X_test = df.loc[df['LABEL_ABOUT'].isnull(), "ABOUT_AUTHOR"]


# labelling y_train:
y_train = mlb.fit_transform(y_train)

# Load model direc:
train_encoding = tokenizer(X_train.tolist(), truncation=True, padding=True)

test_encoding = tokenizer(X_test.tolist(), truncation=True, padding=True)

# Veryi dataframe dönültürmek için


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Verinizi dataset formatına dönüştürün
train_dataset = CustomDataset(train_encoding, y_train)

training_args = TrainingArguments(
    output_dir='./results',  # modelin kaydedileceği dizin
    num_train_epochs=3,  # epoch sayısı
    per_device_train_batch_size=8, # eğitim boyutu
    per_device_eval_batch_size=8, # değerlendirme boyutu
    warmup_steps=500,  # öğrenme oranının yavaşça artırıldığı adım sayısı.
    weight_decay=0.01,  # Ağırlık çürümesi (weight decay) hiperparametresi. Aşırı öğrenmeyi önlemeye yardımcı olur.
    logging_dir='./logs',  # logging_dir: Eğitim loglarının kaydedileceği dizin.
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # X_train ve y_train verilerinizin tokenized dataset versiyonu
)

trainer.train() # Modeli eğittik ve tahmine hazır.

# X_test verisini dataset formatına dönüştürün
test_dataset = CustomDataset(test_encoding, [0] * len(X_test))  # Etiketler bilinmediği için 0 ile dolduruyoruz.
predictions = trainer.predict(test_dataset).predictions

# Tahminleri işleyin
pred_labels = (torch.sigmoid(torch.tensor(predictions)) > 0.5).int()

pred_labels = mlb.inverse_transform(pred_labels.numpy())

# Tahmin edilen etiketleri df'e ekleyin
df.loc[df['LABEL_ABOUT'].isnull(), 'PREDICTED_LABELS'] = pred_labels