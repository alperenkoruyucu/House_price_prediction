# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import locale
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
from scipy import stats
from termcolor import colored
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', None)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
        
data = pd.read_csv("../input/real-estate-in-istanbul-turkey-emlakjet/Real Estate in ISTANBUL (Emlakjet).csv")
datausd = pd.read_csv("/kaggle/input/usdtry/USDTRY.csv")
datausd.info()
data.info()

table_1 = data.describe().T
table_1.style.background_gradient(cmap="ocean_r")
for column in data.select_dtypes(int):
    if column == "Fiyatı":
        q = data[column].quantile(0.99)
        data_outlier =  data[data[column] < q]
    else:
        None
print("--->>> Data has  '",data.shape[0]-data_outlier.shape[0] ,"' Outlier values For PRICE column. <<<<---")

data = data_outlier

data = data.drop(["İlan_Oluşturma_Tarihi"], axis = 1)
data.head()

table_2 = pd.DataFrame(data.nunique())
table_2.style.background_gradient(cmap="Reds")

data2 = pd.DataFrame()

for key in data.keys():
    if data[key].dtype == "object":
        new_key = key + "_cat"
        data2[new_key] = data[key].astype("category").cat.codes
        
    else:
        data2[key] = data[key]

corr = data.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.set(style="dark")
f, ax = plt.subplots(figsize=(14, 12))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.4, center=0,
            square=True, linewidths=.8, cbar_kws={"shrink": .5}, annot=True ,fmt='.1%')

def statistical(df, target):    
    skew = df[target].skew()
    kurtosis = df[target].kurt()
    print(colored("Skewness of data is:","blue"), skew,
                 colored(" >>>>>>>>>>>>> and kurtosis is:","blue"), kurtosis)

statistical(data2,"Fiyatı")

plt.style.use("ggplot")
plt.figure(figsize=(20,8))
plt.title("Distribution of 'House Price' column", size = 30)
data2["Fiyatı"].hist(bins=100,range =(0,10000000), color ="#DF1414")
plt.xlabel("Price", size = 20)
plt.ylabel("Count", size = 20)
plt.show()

y_1 = np.log(data2["Fiyatı"]) 
plt.figure(figsize=(20,8))
plt.title("Distribution of 'House Price' column after log Distribution", color ="red", size = 20)

sns.histplot(data = y_1, bins=100, kde=True, palette="Paired", alpha = 0.4)
plt.xlabel("Price", size = 20)
plt.ylabel("Count", size = 20)
plt.show()

from sklearn.model_selection import train_test_split

X = data2.drop(["Fiyatı"], axis = 1) #Price Column
y = data2["Fiyatı"]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
                                                      test_size=0.25,
                                                      train_size=0.75, 
                                                      shuffle=True,
                                                      random_state=10)


print(" X_train shape is:", X_train.shape)
print(" X_valid shape is:", X_valid.shape)

def log_transformation(data):
    data = np.log(data)
    return data

y_train = log_transformation(y_train)
y_valid = log_transformation(y_valid)

def robust_scale(X_t, X_v):
    
    scaler = RobustScaler()
    return pd.DataFrame(scaler.fit_transform(X_t)), pd.DataFrame(scaler.transform(X_v))

X_train_rs, X_val_rs = robust_scale(X_train, X_valid)

import torch
import torch.nn as nn
# Give a seed for generating reproducible results
torch.manual_seed(0)

class House_Price(nn.Module):
    
    def __init__(self, num_feature):
        # super function. It inherits from nn.Module 
        super().__init__()
        self.layers = nn.Sequential(
            
          nn.Linear(num_feature, 64),
            
          nn.Linear(64, 128),
          #nn.Dropout(0.25),
        
          nn.Linear(128, 32),
          #nn.Dropout(0.25),
     
          nn.Linear(32, 1))
        
    def forward(self, x):
        x = self.layers(x)
        return x.view(-1) 
    
class Emlak_jet_data(torch.utils.data.Dataset):
    
    def __init__(self, X, Y):
        self.X = torch.tensor(X.values,dtype=torch.float)
        self.Y = torch.tensor(Y.values,dtype=torch.float)
        
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
from torch import optim

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"The Model using the {device} device.")
# "***************"

num_feature     = X_train_rs.shape[1]

model          = House_Price(num_feature)
model.to(device)

epochs         = 100
batch_size     = 128

loss_function  = nn.MSELoss()
optimizer      = optim.SGD(model.parameters(), lr=0.00002, weight_decay=1e-10)  #, weight_decay=1e-10
scheduler      = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.4)

print(model)

from torch.utils.data import DataLoader

train_dataset = Emlak_jet_data(X_train_rs, y_train)
valid_dataset = Emlak_jet_data(X_val_rs, y_valid)

train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader  = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# train için shuffle gerekli ancak test için anlamsız.

print(train_dataset.X.shape)
print(valid_dataset.X.shape)

# you can use GPU for model train and predict. Accelerator > GPU
# GPU can decrease the MSELoss
# Than convert the loss lists to numpy

loss_list = []
valid_loss = []
pred_list = []

for epoch in range(epochs):
    
    #Train aşaması
    model.train()
    for x, y in train_loader:
        
        #Modelin tahmin etmesi
        pred_train = model(x.to(device))
        #Loss Fonksiyonu ile hata değerlendirmesi
        loss = loss_function(pred_train, y.to(device))
        # gradları sıfırlıyoruz
        optimizer.zero_grad()
        # backward adımı
        loss.backward()
        #Optimizasyon parametrelerinin yenilenmesi
        optimizer.step()
        
        
    else:
        #Validasyon aşaması
        #Bu blokda, her train döngüsü için optimize edilen ağırlıklar ile validasyonu sağlıyoruz
        model.eval()
        with torch.no_grad():
        
            for x_, y_ in valid_loader:      
                pred = model(x_.to(device))
                loss_valid = loss_function(pred, y_.to(device))
                
        # learning rate parametresinin doğru optimize edildiğini kontrol etmek için onu da yazdıralım.
        if(epoch % 10 == 9): 
            print(colored("Epoch: {}  -  ".format(epoch+1),"red"),
                  colored(">>>>>>> Training MSELoss: {:.5f}   - ".format(loss.data),"green"),
                  colored(">>>>>>> Validation MSELoss: {:.5f}    ".format(loss_valid.data),"green"),
                  colored(">>>>>>> Learning rate: {:.6f}  ".format(scheduler.get_lr()[0]),"blue"))
        
        
    #Her epoch sonrasi train ve validasyon hatalarının kaydedilmesi
    valid_loss.append(loss_valid.data)
    loss_list.append(loss.data)
    
    #Learning rate için belirlediğimiz operasyon parametrelerinin güncellenmesi
    scheduler.step()

print("PyTorch Model Is Completed...")

plt.figure(figsize=(18,6))
plt.plot(range(epochs),valid_loss, color ="red", marker = "o", markersize = 3, label = "Validation Loss")
plt.plot(range(epochs),loss_list, color ="green", marker = "o", markersize = 3, label = "Train Loss")
plt.xlabel("Number of Iterations")
plt.ylabel("Loss")
#plt.axline(15)
plt.legend()
plt.show()


predis = []
X_val_rs = np.array(X_val_rs, dtype=np.float32)
input_x_test = torch.from_numpy(X_val_rs)
#predicted = model(input_x_test.float()).data.numpy()
predis = model(input_x_test)

#from tensor type to numpy
predis_ = pd.DataFrame(predis.cpu().detach().numpy())
predis_.head()

df_predict = pd.DataFrame({'target': predis.cpu().detach().numpy()})
print(df_predict.shape)

df_valid = pd.DataFrame({'target': y_valid})
print(df_valid.shape)

sns.set(style="darkgrid")
plt.figure(figsize=(20,8))
plt.title("Distributions of Log Transformation")
sns.histplot(data = df_valid, bins=100, kde=True, palette="Paired", alpha = 1, label = "validation Values")
sns.histplot(data = df_predict, bins=100, kde=True, palette="rocket", alpha = 0.2, label = "Predicted Values")
plt.xlabel("Price")
plt.grid(True)
plt.legend()
df_valid["target"] = np.exp(df_valid["target"] )
df_predict["target"] = np.exp(df_predict["target"])
print("Validataion \n", statistical(df_valid, "target"))
print("Predicted \n", statistical(df_predict, "target"))

fig, axs = plt.subplots(2,1)
fig.set_figheight(11)
fig.set_figwidth(20)
fig.suptitle('Distributions of Predicted and Validation Values', color = "red")


df_predict.hist(bins=30,range =(0,10000000), color = "#EB60DD", ax = axs[0], label = "Predicted Values")
axs[0].set_xlim([0,10000000])
axs[0].set_ylabel("Count")
axs[0].set_title("")
axs[0].legend()


df_valid.hist(bins=30,range =(0,10000000), color = "#92F05B", ax = axs[1], label = "Validation Values")
axs[1].set_xlim([0,10000000])
axs[1].set_ylabel("Count")
axs[1].legend()
axs[1].set_title("House Prices", color = "red")
plt.show()

data = pd.DataFrame(data)
aylar = {
    'Ocak': '01',
    'Şubat': '02',
    'Mart': '03',
    'Nisan': '04',
    'Mayıs': '05',
    'Haziran': '06',
    'Temmuz': '07',
    'Ağustos': '08',
    'Eylül': '09',
    'Ekim': '10',
    'Kasım': '11',
    'Aralık': '12'
}
data['İlan_Güncelleme_Tarihi'] = data['İlan_Güncelleme_Tarihi'].apply(lambda x: ' '.join([x.split()[0], aylar[x.split()[1]], x.split()[2]]))
print(data['İlan_Güncelleme_Tarihi'])

data['İlan_Güncelleme_Tarihi'] = pd.to_datetime(data['İlan_Güncelleme_Tarihi'], format='%d %m %Y')
datausd['Date'] = pd.to_datetime(datausd['Date'])

def dolar_kuru_ekle(İlan_Numarası):
    ilan_tarihi = data.loc[data['İlan_Numarası'] == İlan_Numarası, 'İlan_Güncelleme_Tarihi'].values[0]
    
    # İlgili tarihi dolar kuru datasetinde ara
    kuru_bul = datausd.loc[datausd['Date'] == ilan_tarihi, 'Close']
    
    # Eğer tarih tam uyuşmuyorsa en yakın tarihli değeri al
    if isinstance(kuru_bul, pd.Series):
        if not kuru_bul.empty:
            return kuru_bul.values[0]
    
    return None

# Her bir ev ilanına dolar kurunu ekleyin
data['dolar_kuru'] = data['İlan_Numarası'].apply(dolar_kuru_ekle)
print(data['dolar_kuru'])