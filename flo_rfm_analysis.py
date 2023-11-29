###############################################################
# 1. İş Problemi (Business Problem)
###############################################################

# FLO müşterilerini segmentlere ayırıp bu segmentlere göre
# pazarlama stratejileri belirlemek istiyor.

# master_id: Eşsiz müşteri numarası
# order_channel: Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
# last_order_channel: En son alışverişin yapıldığı kanal
# first_order_date: Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date: Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online: Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline: Müşterinin offline platformda yaptığı son alışveriş tarihi
##order_num_total_ever_online: Müşterinin online platformda yaptığı toplam alışveriş sayısı
## order_num_total_ever_offline: Müşterinin offline'da yaptığı toplam alışveriş sayısı
## customer_value_total_ever_offline: Müşterinin offline alışverişlerinde ödediği toplam ücret
## customer_value_total_ever_online: Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12: Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi



###############################################################
# Görev 1: Veriyi Anlama ve Hazırlama
###############################################################
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None) '''bu islem tum satirlari da gormek istediginde yapilir.'''
pd.set_option('display.float_format' , lambda x: '%.3f' % x)

df_ = pd.read_csv('/Users/rmarabaci/PycharmProjects/pythonProject1/data_science_bootcamp/Hafta_4_CRM_Analytics/FLO_RFM_Analizi/flo_data_20k.csv')

df = df_.copy()

# Adım2: Veri setinde
# a. İlk 10 gözlem,
# b. Değişken isimleri,
# c. Betimsel istatistik,
# d. Boş değer,
# e. Değişken tipleri, incelemesi yapınız.


def check_df(dataframe, head=10):
    print("#"*9 + " SHAPE-BOYUT-ŞEKİL " + "#"*9)
    print(dataframe.shape)
    print("#"*9 + " DTYPE " + "#"*9)
    print(dataframe.dtypes)
    print("#" * 9 + " HEAD " + "#" * 9)
    print(dataframe.head(head))
    print("#" * 9 + " TAİL " + "#" * 9)
    print(dataframe.tail(head))
    print("#" * 9 + " NA-BOŞ DEĞER " + "#" * 9)
    print(dataframe.isnull().sum())
    print("#" * 9 + " QUANTİLES-NİCELLER " + "#" * 9)
    print(dataframe.describe().T)

check_df(df)


df.columns
df.describe().T
df.isnull().sum()
df.dtypes

#Betimsel Istatistikler - Grafikler
sns.countplot(x=df["order_channel"], data=df)
plt.show()

sns.histplot(x=df["customer_value_total_ever_online"])
plt.show(block=True)

sns.boxplot(x=df["customer_value_total_ever_online"])
plt.show(block=True)

df["customer_value_total_ever_online"].hist()
plt.show()

#Adım3: Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam
#alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

df["order_num_omnichannel"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]

df["customer_value_total_ever_omnichannel"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df.dtypes

#Adım4: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

df['first_order_date'] = pd.to_datetime(df['first_order_date'])

df['last_order_date'] = pd.to_datetime(df['last_order_date'])

df['last_order_date_online'] = pd.to_datetime(df['last_order_date_online'])

df['last_order_date_offline'] = pd.to_datetime(df['last_order_date_offline'])


# Adım5: Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız.
df.head(10)

df['master_id'].value_counts()

df['master_id'].nunique()
df['order_channel'].value_counts()

df.groupby('order_channel').agg({"order_num_omnichannel": ['count', 'sum'],
                         'master_id': 'count',
                         "customer_value_total_ever_omnichannel":['sum', 'mean']})

df['master_id'].describe().T

df['customer_value_total_ever_omnichannel'].describe().T

df['order_num_omnichannel'].describe().T

# Adım6: En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.

df.sort_values('customer_value_total_ever_omnichannel', ascending=False).head(10)

# Adım7: En fazla siparişi veren ilk 10 müşteriyi sıralayınız.

df.sort_values('order_num_omnichannel', ascending=False).head(10)

# Adım8: Veri ön hazırlık sürecini fonksiyonlaştırınız.

def data_prep(dataframe):
    dataframe["order_num_omnichannel"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total_ever_omnichannel"] = dataframe["customer_value_total_ever_offline"] + dataframe[
        "customer_value_total_ever_online"]

    df['first_order_date'] = pd.to_datetime(df['first_order_date'])

    df['last_order_date'] = pd.to_datetime(df['last_order_date'])

    df['last_order_date_online'] = pd.to_datetime(df['last_order_date_online'])

    df['last_order_date_offline'] = pd.to_datetime(df['last_order_date_offline'])

    return dataframe

###############################################################
# Görev 2: RFM Metriklerinin Hesaplanması
###############################################################

# Adım 1: Recency, Frequency ve Monetary tanımlarını yapınız.

 # Recency = analizin yapildigi tarih - ilgili musterinin son satin alma yaptigi tarih
 # Frequency = Musterinin yaptigi toplam satin alma miktari
 # Monetary = musterinin yaptigi toplam satin almalar neticesinde biraktigi toplam parasal deger


# Adım 2: Müşteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayınız.
# Adım 3: Hesapladığınız metrikleri rfm isimli bir değişkene atayınız.
# Adım 4: Oluşturduğunuz metriklerin isimlerini recency, frequency ve monetary olarak değiştiriniz.
df.head()

df['last_order_date_offline'].max()
df['last_order_date_online'].max()

today_date = dt.datetime(2021, 6, 1)
# Bonus Bilgi: type = dt.datetime(2021, 6, 1, format='%d/%m/%Y') Bu sekilde yaparak formati istedigimiz sekilde ayarlayabniliriz
type(today_date)

df['general_order_date'] = df[['last_order_date_offline', 'last_order_date_online']].max(axis=1)

rfm = df.groupby('master_id').agg({'general_order_date': lambda general_order_date : (today_date - general_order_date.max()).days,
                                     'order_num_omnichannel': lambda order_num_omnichannel : order_num_omnichannel,
                                     'customer_value_total_ever_omnichannel': lambda customer_value_total_ever_omnichannel : customer_value_total_ever_omnichannel})

rfm.columns = ['recency', 'frequency', 'monetary']

rfm.head()

rfm.shape

###############################################################
# Görev 3: RF Skorunun Hesaplanması
###############################################################

# Adım 1: Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz.
# Adım 2: Bu skorları recency_score, frequency_score ve monetary_score olarak kaydediniz.

rfm['recency_score'] = pd.qcut(rfm['recency'] , 5, labels= [5,4,3,2,1])

rfm['monetary_score'] = pd.qcut(rfm['monetary'] , 5, labels= [1,2,3,4,5])

rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method= 'first'), 5, labels= [1,2,3,4,5])


# Adım 3: recency_score ve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.

rfm['RFM_SCORE'] = rfm['recency_score'].astype(str) \
                   + rfm['frequency_score'].astype(str)


###############################################################
# Görev 4: RF Skorunun Segment Olarak Tanımlanması
###############################################################

# Adım 1: Oluşturulan RF skorları için segment tanımlamaları yapınız.

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

# Adım 2: Aşağıdaki seg_map yardımı ile skorları segmentlere çeviriniz.

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex = True)


###############################################################
# Görev 5: Aksiyon Zamanı !
###############################################################

# Adım1: Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.

rfm[['segment', 'recency', 'frequency' , 'monetary']].groupby('segment').agg(['mean' ,'count'])

# Adım2: RFM analizi yardımıyla aşağıda verilen 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv olarak kaydediniz.

# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri
# tercihlerinin üstünde. Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak
# iletişime geçmek isteniliyor. Sadık müşterilerinden(champions, loyal_customers) ve kadın kategorisinden alışveriş
# yapan kişiler özel olarak iletişim kurulacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına kaydediniz.

rfm["customer_id"] = df["master_id"]
target_segments_customer_ids = rfm[rfm["segment"].isin(["champions", "loyal_customers"])]["customer_id"]
cust_ids = \
df[(df["master_id"].isin(target_segments_customer_ids)) & (df["interested_in_categories_12"].str.contains("KADIN"))][
    "master_id"]
cust_ids.to_csv("yeni_marka_hedef_müşteri_id.csv", index=False)
cust_ids.shape

rfm.head()




# b. Erkek ve Çocuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte
# iyi müşteri olan ama uzun süredir alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni
# gelen müşteriler özel olarak hedef alınmak isteniyor. Uygun profildeki müşterilerin id'lerini csv dosyasına kaydediniz.

target_segments_customer_ids = rfm[rfm["segment"].isin(["cant_loose", "hibernating", "new_customers"])]["customer_id"]
cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) & (
            (df["interested_in_categories_12"].str.contains("ERKEK")) | (df["interested_in_categories_12"].str.contains("COCUK")))]["master_id"]
# cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) & (df["interested_in_categories_12"].str.contains("ERKEK|COCUK"))]["master_id"]

cust_ids.to_csv("indirim_hedef_müşteri_ids.csv", index=False)

cust_ids.shape
