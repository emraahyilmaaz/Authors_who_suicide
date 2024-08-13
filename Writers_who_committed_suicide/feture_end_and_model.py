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



import pandas as pd

df = pd.read_csv("/Users/emrahyilmaz/Desktop/my_model_label.csv", delimiter=";", header=1)
df = df.dropna(subset=['ABOUT_AUTHOR'])

df.to_csv('AUTHOURS.csv')

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










import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import warnings
import re
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import random
from matplotlib.colors import CSS4_COLORS

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.multiclass')
warnings.simplefilter(action="ignore")
warnings.simplefilter(action="ignore")

pd.set_option("display.expand_frame_repr", False)

df = pd.read_csv('collect_data/web_scraping/my_model_label.csv', delimiter=";", header=1)
df = df.dropna(subset=['ABOUT_AUTHOR'])

#df.drop(['BOOK_1', 'BOOK_2', 'BOOK_3','PREDICTED_LABEL_1', 'PREDICTED_LABEL_2', 'PREDICTED_LABEL_3',
          #'PREDICTED_LABEL_4','PREDICTED_LABEL_5','PREDICTED_LABEL_6', 'PREDICTED_LABEL_7', 'PREDICTED_LABEL_8',
          #'PREDICTED_LABEL_9', 'PREDICTED_LABEL_10', 'PREDICTED_LABELS'], inplace=True, axis=1)


def cleaninf_text_df(text):
    text = str(text)
    text = text.lower()
    # Özel karakterleri ve noktalama işaretlerini kaldırma
    text = re.sub(r'[^\w\s]', '', text)
    # Gereksiz boşlukları kaldırma
    text = re.sub(r'\s+', ' ', text).strip()
    return text


df['ABOUT_AUTHOR'] = df['ABOUT_AUTHOR'].apply(lambda x: cleaninf_text_df(x))



##################
# PROCCESSİNG DATA
##################

def keyword_check(text, keywords):
    if not isinstance(text, str):  # Eğer text string değilse
        return 0
    if isinstance(keywords, list):  # Eğer anahtar kelimeler bir liste ise
        for keyword in keywords:
            if keyword.lower() in text.lower():
                return 1
        return 0
    else:  # Eğer tek bir anahtar kelime ise
        if keywords.lower() in text.lower():
            return 1
        else:
            return 0


drug = ['uyuşturucu', 'kumar', "madde bağımlılığı", "morfin", "alkol", 'ilac']
# depend
df['DEPEND'] = df['ABOUT_AUTHOR'].apply(lambda x: keyword_check(x, drug))

# poem:
df['JOURNALIST'] = df['ABOUT_AUTHOR'].apply(lambda x: keyword_check(x, 'gazeteci'))


# education
univercity = ['üniversite', 'universite', 'akademisyen', 'profesör', 'docent', 'öğretmen']
df['HIGH SCHOOL'] = df['ABOUT_AUTHOR'].apply(lambda x: keyword_check(x, univercity))


# Bilim kurgu:
science_fiction = ['kurgu', 'bilim kurgu', 'hayal', 'distopik', 'hayal']
df['SCİENCE_FİCTİON'] = df['ABOUT_AUTHOR'].apply(lambda x: keyword_check(x, science_fiction))

# Fantezi:
fantasy = ['mitoloji', 'fantezi', 'polisiye']
df['FANTASY'] = df['ABOUT_AUTHOR'].apply(lambda x: keyword_check(x, fantasy))

# rich:
rich = ['zengin', 'varlıklı']
df['RICH'] = df['ABOUT_AUTHOR'].apply(lambda x: keyword_check(x, rich))

# Store:
store = ['hikaye', 'öykü', 'kısa roman']
df['STORE'] = df['ABOUT_AUTHOR'].apply(lambda x: keyword_check(x, store))

# Felsefe:
felsefe = ['felsefe', 'Absürdizm', 'varoluşçuluk', 'sosyolog', 'toplumbilimci', 'toplum', 'birey', 'politik',
           'fikir adamı', 'filozof', 'sosyoloji']
df['FELSEFE'] = df['ABOUT_AUTHOR'].apply(lambda x: keyword_check(x, felsefe))

# pskoloji
pskoloji = ['Psikoloji', 'psikiatrist', 'nihilizm', 'pozitivizm']
df['PSYCHOLOGY'] = df['ABOUT_AUTHOR'].apply(lambda x: keyword_check(x, pskoloji))

# art
Art = ['sanat', 'resim', 'ressam', 'müzik', 'güzel sanatlar', 'mimar', 'reklam']
df['ART'] = df['ABOUT_AUTHOR'].apply(lambda x: keyword_check(x, Art))

# siyasi baskı
politic_press = ['Siyasi baskı', 'idam edildi', 'rejim karşıtı', 'aktivist', 'tutuklandı', 'komünist rejim',
                 'faşist', 'kaçtı', 'baskısından', 'sürgün', 'nazi', 'rejim', 'baskı', 'kaçmak zorunda kaldı',
                 'kaçarken', 'yönetimle savaştı']
df['POLİTİC_PRESS'] = df['ABOUT_AUTHOR'].apply(lambda x: keyword_check(x, politic_press))

# siyasetçi
politic = ['vezir', 'devlet adamı', 'siyaset felsefesi', 'siyaset', 'milletvekili', 'anayasa', 'seçim']
df['POLİTİC'] = df['ABOUT_AUTHOR'].apply(lambda x: keyword_check(x, politic))

# savaş mağduru
war = ['i savaşı', 'ii savaşı', 'iç savaş', 'kaçtı', 'kaçarken', 'savaş', 'orduya katıldı','savaşı',
       'birinci dünya savaşı', 'ikinci dünya savaşı', 'yaralandı', 'savaştı']
df['WAR'] =  df['ABOUT_AUTHOR'].apply(lambda x: keyword_check(x, war))

# çocuk yazarı
child = ['çocuk kitabı', 'çocuk yazarı', 'çocuk hikayeleri', 'çocuk yazarı', 'çocuk yazarı']
df['CHİLD_STORE'] = df['ABOUT_AUTHOR'].apply(lambda x: keyword_check(x, child))

# medyatil
radio_tv_magazine = ['dergi', 'televizyon', 'radyo', 'yönetmen', 'çekim'
                     'program', 'programı', 'belgesel', 'köşe yazarlığı', 'film', 'metraj']
df['RADİO_TV_MAGAZİNE'] = df['ABOUT_AUTHOR'].apply(lambda x: keyword_check(x, radio_tv_magazine))

# mülteci
df['MİGRATİON'] = df['ABOUT_AUTHOR'].apply(lambda x: keyword_check(x, 'göç'))

# mutsuz hayat
unhapy = ['trajik', 'depresyon', 'depresyonu', 'mutsuz', 'şiddet', 'manikdepresif', 'akıl hastası',
        'akıl hastanesi', 'kaybetti', 'boşandı', 'babasının ölümü', 'annesinin ölümü', 'boşanması']
df['UNHAPPY'] = df['ABOUT_AUTHOR'].apply(lambda x: keyword_check(x, unhapy))

# Tarihçi
history = ['tarih']
df['HİSTORY'] = df['ABOUT_AUTHOR'].apply(lambda x: keyword_check(x, history))

# dini
islam = ['Din adamı', 'islam', 'kuran', 'muhammed', 'iman']
df['RELİGİON'] = df['ABOUT_AUTHOR'].apply(lambda x: keyword_check(x, islam))

# Fakir
poor = ['küçükken çalışmak zorunda kaldı', 'fakir', 'yoksul', 'bakmak zorunda kaldı']
df['POOR'] = df['ABOUT_AUTHOR'].apply(lambda x: keyword_check(x, poor))

# tiyatrocu
tiyatro = ['oyuncu', 'sahne', 'skeç', 'oyun', 'tiyatro eseri', 'oyun yazarı']
df['THEATER'] = df['ABOUT_AUTHOR'].apply(lambda x: keyword_check(x, tiyatro))

# Bilim insanı:
scientist = ['Bilim adamı', 'makale', 'araştırma görevlisi', 'doçent']
df['SCİENTİST'] = df['ABOUT_AUTHOR'].apply(lambda x: keyword_check(x, scientist))

# Mativasyon:
motivation = ['motisvasyon konuşmacısı', 'konuşmacı', 'kişisel gelişim', 'yaşam koçu']

# commentleri ve linkleri çıkar:
df.head()

df = df.drop(['LINKS', 'ABOUT_AUTHOR', 'COMMENT', 'BOOK_1', 'BOOK_2', 'BOOK_3', 'BORN', 'DEATH',
              'BORN_COUNTRY', 'LABEL_ABOUT'], axis=1)


df['AVR_SENTİMENT'] = (df['ABOUT_SENTIMENT'] + df['BOOK1_SENTIMENT'] + df['BOOK2_SENTIMENT'] +
                       df['BOOK3_SENTIMENT'] + df['COMMENT_sentiment_average'])


df = df.drop(['ABOUT_SENTIMENT', 'BOOK1_SENTIMENT', 'BOOK2_SENTIMENT',
              'BOOK3_SENTIMENT', 'COMMENT_sentiment_average'], axis=1)

# Doğum tarihlerine göre kategorile:


df['CATEGORY_BORN'] = pd.cut(df['BORN_YEAR'], bins=[0, 1000, 1880, 1930, 2020],
                             labels=['Antik', 'clasic', 'war', 'modern'])

df = df.drop('BORN_YEAR', axis=1)

scaler = MinMaxScaler()
df[['RATE', 'READ', 'LIKED', 'SEEN']] = scaler.fit_transform(df[['RATE', 'READ', 'LIKED', 'SEEN']])

weights = {
    'RATE': 0.1,  # %10 etkili
    'READ': 0.3,  # %30 etkili
    'LIKED': 0.3, # %30 etkili
    'SEEN': 0.3   # %30 etkili
}

# Ağırlıklı popülerlik puanı oluşturma
df['POPULARITY_SCORE'] = (
    df['RATE'] * weights['RATE'] +
    df['READ'] * weights['READ'] +
    df['LIKED'] * weights['LIKED'] +
    df['SEEN'] * weights['SEEN']
)

df = df.drop(['RATE', 'READ', 'LIKED', 'SEEN'], axis=1 )

df.to_csv('Author_dataset.csv')
##########################
# EXPLORATORY DATA ANALYSİS
##########################

Train = df.loc[df['SUİCEDE'].notnull()]
Test = df.loc[df['SUİCEDE'].isnull()]

def grap_col_names(dataframe, cat_th=10, car_th=20):
    """

       Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
       Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

       Parameters
       ------
           dataframe: dataframe
                   Değişken isimleri alınmak istenilen dataframe
           cat_th: int, optional
                   numerik fakat kategorik olan değişkenler için sınıf eşik değeri
           car_th: int, optinal
                   kategorik fakat kardinal değişkenler için sınıf eşik değeri

       Returns
       ------
           cat_cols: list
                   Kategorik değişken listesi
           num_cols: list
                   Numerik değişken listesi
           cat_but_car: list
                   Kategorik görünümlü kardinal değişken listesi

       Examples
       ------
           import seaborn as sns
           df = sns.load_dataset("iris")
           print(grab_col_names(df))


       Notes
       ------
           cat_cols + num_cols + cat_but_car = toplam değişken sayısı
           num_but_cat cat_cols'un içerisinde.
           Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

       """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]  # Kategorik kolonlar
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]  # Sayısal tipte ama kategorik değişken
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]  # String ifade ama yüksek sınıflı (isimler gibi)

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num cols:
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"observation: {dataframe.shape[0]}")
    print(f"Variable: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grap_col_names(df)


y_train = Train["SUİCEDE"]
X_train = Train.drop(["SUİCEDE", "NAME", "CATEGORY_BORN"], axis=1)


def create_pca_df(X, y):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(X)
    pca_df = pd.DataFrame(pca_fit, columns=["PCA1", "PCA2"])
    final_df = pd.concat([pca_df, pd.DataFrame(y)], axis=1)
    return final_df


final_df = create_pca_df(X_train, y_train)


def plot_pca(dataframe, target):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("PC1", fontsize=15)
    ax.set_ylabel("PC2", fontsize=15)
    ax.set_title(f"{target.capitalize()}", fontsize=20)

    color_list = list(CSS4_COLORS.keys())
    targets = list(dataframe[target].unique())
    colors = random.sample(color_list, len(targets))

    for t, color in zip(targets, colors):
        indices = dataframe[target] == t
        ax.scatter(dataframe.loc[indices, 'PCA1'], dataframe.loc[indices, 'PCA2'], c=color, s=50, label=t)
    ax.legend()
    ax.grid()
    plt.show()


plot_pca(final_df, 'SUİCEDE')


def target_summary_with_cat(dataframe, target, categorical_col):
    # Değer sayısını ve ortalama hedef değişkeni hesaplama
    count_series = dataframe[categorical_col].value_counts()
    mean_series = dataframe.groupby(categorical_col)[target].mean()

    # DataFrame oluşturma
    summary_df = pd.DataFrame({
        categorical_col: count_series.index,
        "Count": count_series.values,
        "Target_Mean": mean_series.values
    })

    # İlgili sütunu index olarak ayarlama
    summary_df.set_index(categorical_col, inplace=True)

    print(summary_df, end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(Train, 'SUİCEDE', col )


Train.groupby('SUİCEDE')['UNHAPPY'].sum()


def target_summary_with_cat(dataframe, target, categorical_col):
    # Değer sayısını ve ortalama hedef değişkeni hesaplama
    count_series = dataframe[categorical_col].value_counts()
    mean_series = dataframe.groupby(categorical_col)[target].mean()

    # DataFrame oluşturma
    summary_df = pd.DataFrame({
        categorical_col: count_series.index,
        "Count": count_series.values,
        "Target_Mean": mean_series.values
    })

    # İlgili sütunu index olarak ayarlama
    summary_df.set_index(categorical_col, inplace=True)

    print(summary_df, end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(Train, 'SUİCEDE', col)


def plot_suicide_rate_for_category(dataframe, target, categorical_col, category_value=1, save_path=None):
    # Sadece belirli bir sınıfı (category_value) filtreleme
    filtered_df = dataframe[dataframe[categorical_col] == category_value]

    # Eksik verileri kontrol etme
    missing_suicide = filtered_df[target].isna().sum()
    print(f'Filtrelenmiş veri setinde {missing_suicide} adet eksik SUİCEDE değeri var.')

    # Eksik verileri hariç tutma
    filtered_df = filtered_df.dropna(subset=[target])

    # SUİCEDE değerine göre sayımları hesaplama
    suicide_counts = filtered_df[target].value_counts()

    # Toplam kişi sayısı
    total_count = len(filtered_df)

    if total_count == 0:
        print(f"{categorical_col} = {category_value} için veri yok.")
        return

    # Pasta grafiği çizimi
    plt.figure(figsize=(8, 8))

    # Renkleri özelleştirme (Turkuaz ve Koyu Mavi)
    colors = ['#40E0D0', '#1E90FF']  # Turquoise ve DodgerBlue

    if len(suicide_counts) == 1:
        if 1 in suicide_counts.index:
            labels = [f'İntihar Edenler: {suicide_counts[1]}', 'İntihar Etmeyenler: 0']
            plt.pie([suicide_counts[1], 0], labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
        else:
            labels = ['İntihar Edenler: 0', f'İntihar Etmeyenler: {suicide_counts[0]}']
            plt.pie([0, suicide_counts[0]], labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    else:
        labels = [f'İntihar Edenler: {suicide_counts.get(1, 0)}', f'İntihar Etmeyenler: {suicide_counts.get(0, 0)}']
        plt.pie(suicide_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)

    plt.title(f'{categorical_col} = {category_value} - Toplam: {total_count}')

    # Grafik kaydetme
    if save_path:
        plt.savefig(save_path)
        print(f'Grafik kaydedildi: {save_path}')

    plt.show()



for col in cat_cols:
    file_name = f"{col}_suicide_distribution.png"
    save_path = f"./{file_name}"  # Kaydedilecek dosya yolu
    plot_suicide_rate_for_category(df, 'SUİCEDE', col, category_value=1, save_path=save_path)

###################
# FEATURE ENGİNEERİNG
###################

# onehot encoding:
df = pd.get_dummies(df, columns=['CATEGORY_BORN'], dtype=int, drop_first=True)

# scaler
sc = MinMaxScaler()
df['AVR_SENTİMENT'] = sc.fit_transform(df[['AVR_SENTİMENT']])

#############
# MODEL KURMA
#############
Train = df.loc[df['SUİCEDE'].notnull()]
X_train = Train.drop(['SUİCEDE', 'NAME'], axis=1)
y_train = Train['SUİCEDE']


def base_model(X, y, scoring='f1'):
    print('Base Models..... ')
    classification = [
        ('LR', LogisticRegression(max_iter=1000, class_weight='balanced')),
        ('KNN', KNeighborsClassifier()),
        ('CART', DecisionTreeClassifier(class_weight='balanced')),
        ('RF', RandomForestClassifier(class_weight='balanced')),
        ('GBM', GradientBoostingClassifier()),
        ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                                  scale_pos_weight=len(y[y == 0]) / len(y[y == 1]))),
        ("LightGBM", LGBMClassifier(verbose=-1, class_weight='balanced'))
    ]


    for name, classifier in classification:
        try:
            scores = cross_val_score(classifier, X, y, cv=5, scoring=scoring)
            mean_score = np.mean(scores)
            print(f"f1: {round(mean_score, 4)} ({name}) ")
        except Exception as e:
            print(f"Model {name} hata verdi: {e}")


base_model(X_train, y_train)


cart_params = {'max_depth': range(1, 20),
               'min_samples_split': range(2, 30)}

xgboost_params = {'learning_rate': [0.1, 0.01],
                  'max_depth': [5, 8],
                  'n_estimators': [100, 200, 300]}

lightgbm_params = {'learning_rate': [0.1, 0.01],
                   'n_estimators': [300, 500, 700]}



classification = [
    ("XGBoost", XGBClassifier(), xgboost_params),
    ('CART', DecisionTreeClassifier(class_weight='balanced'), cart_params),
    ("LightGBM", LGBMClassifier(verbose=-1, class_weight='balanced'), lightgbm_params)]


def hyperparameter_optimization(X, y, cv=5, scoring='f1'):
    print("Hyperparameters optimization...")
    best_model = {}
    classification = [
        ("XGBoost", XGBClassifier(use_label_encoder=False,
                                  scale_pos_weight=len(y[y == 0]) / len(y[y == 1])),
         xgboost_params),
        ('CART', DecisionTreeClassifier(class_weight='balanced'),
         cart_params),
        ("LightGBM", LGBMClassifier(verbose=-1, class_weight='balanced'),
         lightgbm_params)
    ]

    for name, classifier, params in classification:
        print(f'############# {name} #############')
        cv_results = cross_val_score(classifier, X, y, cv=cv, scoring=scoring)
        print(f'{scoring} (Before): {round(cv_results.mean(), 4)}')

        # GridSearchCV'de varsayılan parametrelerle aynı sonuçları alabilmek için parametre aralıklarına varsayılanları da ekleyin
        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = gs_best.best_estimator_

        cv_results = cross_val_score(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results.mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_model[name] = final_model

    return best_model


# Hiperparametre optimizasyonu
best_model = hyperparameter_optimization(X_train, y_train)





# Voting:
def voting_classifier(best_model, X, y):
    print("Voting classifier....")

    voting_clf = VotingClassifier(estimators=[('CART', best_model["CART"]),
                                               ('LightGBM', best_model['LightGBM']),
                                              ("XGBoost", best_model["XGBoost"])], voting='soft').fit(X, y)

    cv_result = cross_validate(voting_clf, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
    print(f"accuracy: {cv_result['test_accuracy'].mean()}")
    print(f"f1 : {cv_result['test_f1'].mean()}")
    print(f"roc_auc: {cv_result['test_roc_auc'].mean()}")
    return voting_clf


voting_clf = voting_classifier(best_model, X_train, y_train)




X_test = df.loc[df['SUİCEDE'].isnull()]

X_test_pre = X_test.drop(['NAME', 'SUİCEDE'], axis=1)

intihar_riski = voting_clf.predict_proba(X_test_pre)


X_test['İNTİHAR ETME RİSKİ (YÜZDE OLARAK)'] = (intihar_riski[:, 1] * 100).round(2)


Result = X_test[['NAME', 'İNTİHAR ETME RİSKİ (YÜZDE OLARAK)']]

# ufak değerleri çıkar:
X_test_pre['keyword_count'] = X_test_pre[['DEPEND', 'JOURNALIST', 'HIGH SCHOOL',
       'SCİENCE_FİCTİON', 'FANTASY', 'RICH', 'STORE', 'FELSEFE',
       'PSYCHOLOGY', 'ART', 'POLİTİC_PRESS', 'POLİTİC', 'WAR', 'CHİLD_STORE',
       'RADİO_TV_MAGAZİNE', 'MİGRATİON', 'UNHAPPY', 'HİSTORY', 'RELİGİON',
       'POOR', 'THEATER', 'SCİENTİST']].sum(axis=1)

X_test_filtered = X_test_pre[X_test_pre['keyword_count'] >= 4]

X_test_filtered = X_test_filtered.drop(columns=['keyword_count'])

intihar_riski_filtered = voting_clf.predict_proba(X_test_filtered)

# Tahmin edilen intihar riskini yüzdesel olarak hesaplama
X_test.loc[X_test_filtered.index, 'İNTİHAR ETME RİSKİ'] = (intihar_riski_filtered[:, 1] * 100).round(1)
X_test_filtered['İNTİHAR ETME RİSKİ (YÜZDE OLARAK)'] = (intihar_riski_filtered[:, 1] * 100).round(2)

# X_test DataFrame'inden NAME sütununu X_test_filtered'a ekleme
X_test_filtered['NAME'] = X_test.loc[X_test_filtered.index, 'NAME']


df = X_test_filtered[['NAME','İNTİHAR ETME RİSKİ (YÜZDE OLARAK)']]
df = df.sort_values(by='İNTİHAR ETME RİSKİ (YÜZDE OLARAK)', ascending=False)

df.to_csv('FİNAL_MODEL_II.csv', index=False)

