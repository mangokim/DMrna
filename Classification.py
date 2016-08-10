
# coding: utf-8

# In[70]:

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression as lr
import csv
import pprint
from sklearn.neural_network import *
print ("hello")


# In[3]:

#f = open('Full Set.csv')
#csv_f = csv.reader(f)
#for row in csv_f:
#    print (row)
#    if row[2] == 'NULL':
#        print ("NULL")


# In[4]:

df=pd.read_csv('Training Set 4 FULL.csv', sep=',',header=0)
df.values


# In[5]:

var_mod = ['Base','CompBase']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i].factorize()[0])
df.values
df.shape


# In[6]:

train_cols = df.columns[0:7]
# Index([gre, gpa, prestige_2, prestige_3, prestige_4], dtype=object)

logit = sm.Logit(df['Helix'], df[train_cols])

# fit the model
result = logit.fit()


# In[7]:

print (result.summary())


# In[8]:

traindata = df.columns[0:8]
traintarget = df.columns[8]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(df[traindata], df['Helix'])


# In[9]:

from IPython.display import Image 
from sklearn.externals.six import StringIO
import pydot
with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
#import os
#os.unlink('iris.dot')
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("iris.pdf")  


# In[10]:

dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data,  
                     feature_names=traindata,  
                     class_names=traintarget,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png()) 


# In[11]:

def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("iris.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")


# In[12]:

visualize_tree(clf, traindata)


# In[13]:

ndf=pd.read_csv('B Burgdorferi Helix TREE TEST 2.csv', sep=',',header=0)
nvar_mod = ['Base', 'CompBase']
nle = LabelEncoder()
for i in nvar_mod:
    ndf[i] = nle.fit_transform(ndf[i].factorize()[0])
ntraindata = ndf.columns[0:8]
clf.predict(ndf[ntraindata])


# In[14]:

ntraindata


# In[15]:

df=pd.read_csv('Training Set 4 FULL no pairs.csv', sep=',',header=0)
df.values
var_mod = ['Base']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i].factorize()[0])
df.values
df.shape
traindata = df.columns[0:7]
traintarget = df.columns[7]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(df[traindata], df['Helix'])
ndf=pd.read_csv('B Burgdorferi Helix TREE TEST 2 no pairs.csv', sep=',',header=0)
nvar_mod = ['Base']
nle = LabelEncoder()
for i in nvar_mod:
    ndf[i] = nle.fit_transform(ndf[i].factorize()[0])
ntraindata = ndf.columns[0:7]
clf.predict(ndf[ntraindata])


# In[16]:

from sklearn.neural_network import *
X = traindata
y = traintarget
mlf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
mlf.fit(X, y) 
MLPClassifier(activation='relu', algorithm='l-bfgs', alpha=1e-05,
       batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)


# In[17]:

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
get_ipython().magic('matplotlib inline')
dta = sm.datasets.fair.load_pandas().data
dta.educ.hist()
plt.title('Histogram of Education')
plt.xlabel('Education Level')
plt.ylabel('Frequency')


# In[18]:

dftraindata = pd.DataFrame(df[traindata])
dftraindata


# In[19]:

dftraindata.Base.hist()


# In[20]:

model = LogisticRegression()
model = model.fit(dftraindata, df['Helix'])
model.score(dftraindata, df['Helix'])


# In[21]:

model.coef_


# In[22]:

from sklearn.naive_bayes import BernoulliNB, MultinomialNB


# In[23]:

from sklearn import svm
X = df[traindata]
y = df['Helix']
clf = svm.SVC()
clf.fit(X, y)  


# In[24]:

clf.support_vectors_


# In[25]:

ndf=pd.read_csv('B Burgdorferi Helix TREE TEST 2 no pairs WIPED.csv', sep=',',header=0)
nvar_mod = ['Base']
nle = LabelEncoder()
for i in nvar_mod:
    ndf[i] = nle.fit_transform(ndf[i].factorize()[0])
ntraindata = ndf.columns[0:7]
clf.predict(ndf[ntraindata])


# In[26]:

strand = 'AAAUUGAAGAGUUUGAUCAUGGCUCAGAUUGAACGCUGGCGGCAGGCCUAACACAUGCAAGUCGAACGGUAACAGGAAGAAGCUUGCUUCUUUGCUGACGAGUGGCGGACGGGUGAGUAAUGUCUGGGAAACUGCCUGAUGGAGGGGGAUAACUACUGGAAACGGUAGCUAAUACCGCAUAACGUCGCAAGACCAAAGAGGGGGACCUUCGGGCCUCUUGCCAUCGGAUGUGCCCAGAUGGGAUUAGCUAGUAGGUGGGGUAACGGCUCACCUAGGCGACGAUCCCUAGCUGGUCUGAGAGGAUGACCAGCCACACUGGAACUGAGACACGGUCCAGACUCCUACGGGAGGCAGCAGUGGGGAAUAUUGCACAAUGGGCGCAAGCCUGAUGCAGCCAUGCCGCGUGUAUGAAGAAGGCCUUCGGGUUGUAAAGUACUUUCAGCGGGGAGGAAGGGAGUAAAGUUAAUACCUUUGCUCAUUGACGUUACCCGCAGAAGAAGCACCGGCUAACUCCGUGCCAGCAGCCGCGGUAAUACGGAGGGUGCAAGCGUUAAUCGGAAUUACUGGGCGUAAAGCGCACGCAGGCGGUUUGUUAAGUCAGAUGUGAAAUCCCCGGGCUCAACCUGGGAACUGCAUCUGAUACUGGCAAGCUUGAGUCUCGUAGAGGGGGGUAGAAUUCCAGGUGUAGCGGUGAAAUGCGUAGAGAUCUGGAGGAAUACCGGUGGCGAAGGCGGCCCCCUGGACGAAGACUGACGCUCAGGUGCGAAAGCGUGGGGAGCAAACAGGAUUAGAUACCCUGGUAGUCCACGCCGUAAACGAUGUCGACUUGGAGGUUGUGCCCUUGAGGCGUGGCUUCCGGAGCUAACGCGUUAAGUCGACCGCCUGGGGAGUACGGCCGCAAGGUUAAAACUCAAAUGAAUUGACGGGGGCCCGCACAAGCGGUGGAGCAUGUGGUUUAAUUCGAUGCAACGCGAAGAACCUUACCUGGUCUUGACAUCCACGGAAGUUUUCAGAGAUGAGAAUGUGCCUUCGGGAACCGUGAGACAGGUGCUGCAUGGCUGUCGUCAGCUCGUGUUGUGAAAUGUUGGGUUAAGUCCCGCAACGAGCGCAACCCUUAUCCUUUGUUGCCAGCGGUCCGGCCGGGAACUCAAAGGAGACUGCCAGUGAUAAACUGGAGGAAGGUGGGGAUGACGUCAAGUCAUCAUGGCCCUUACGACCAGGGCUACACACGUGCUACAAUGGCGCAUACAAAGAGAAGCGACCUCGCGAGAGCAAGCGGACCUCAUAAAGUGCGUCGUAGUCCGGAUUGGAGUCUGCAACUCGACUCCAUGAAGUCGGAAUCGCUAGUAAUCGUGGAUCAGAAUGCCACGGUGAAUACGUUCCCGGGCCUUGUACACACCGCCCGUCACACCAUGGGAGUGGGUUGCAAAAGAAGUAGGUAGCUUAACCUUCGGGAGGGCGCUUACCACUUUGUGAUUCAUGACUGGGGUGAAGUCGUAACAAGGUAACCGUAGGGGAACCUGCGGUUGGAUCACCUCCUUA'           


# In[27]:

lstrand=list(strand)


# In[28]:

lstrand


# In[29]:

with open('E Coli.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #for i in lstrand:
    spamwriter.writerow([lstrand])


# In[30]:

df=pd.read_csv('E Coli.csv', sep=',',header=0)


# In[31]:

df


# In[32]:

with open("E Coli ALPS.txt", "r") as ins:
    array = []
    for line in ins:
        array.append(line)


# In[33]:

array


# In[34]:

with open("E Coli ALPS Stripped.txt") as textFile:
    lines = [line.split() for line in textFile]


# In[35]:

lines


# In[36]:

lines[5][0]


# In[37]:

len(lines)


# In[38]:

dataArray = []
hpArray = []
h5Array = []
h3Array = []
msArray = []
iArray = []
bArray = []
fArray = []
tArray = []
strandArray = pd.DataFrame(columns=['NucNum','Base','Hairpin','Helix5','Helix3','Multistem','Internal','Bulge','Free','Tail'])
for z in range (1, 1543):
    dataArray.append(z)
    tArray.append('FALSE')
    fArray.append('FALSE')
    bArray.append('FALSE')
    iArray.append('FALSE')
    msArray.append('FALSE')
    h3Array.append('FALSE')
    h5Array.append('FALSE')
    hpArray.append('FALSE')
for x in range (0, 397):
    lower = lines[x][0]
    lower = int(lower) - 1
    upper = lines[x][1]
    upper = int(upper)
    formation = lines[x][2]
    if formation == 'TAIL':
        for i in range (lower, upper):
            tArray[i] = 'TRUE'
    elif formation == 'FREE':
        for i in range (lower, upper):
            fArray[i] = 'TRUE'
    elif formation == 'BULGE':
        for i in range (lower, upper):
            bArray[i] = 'TRUE'
    elif formation == 'INTERNAL':
        for i in range (lower, upper):
            iArray[i] = 'TRUE'
    elif formation == 'MULTISTEM':
        for i in range (lower, upper):
            msArray[i] = 'TRUE'
    elif formation == 'HELIX3':
        for i in range (lower, upper):
            h3Array[i] = 'TRUE'
    elif formation == 'HELIX5':
        for i in range (lower, upper):
            h5Array[i] = 'TRUE'
    elif formation == 'HAIRPIN':
        for i in range (lower, upper):
            hpArray[i] = 'TRUE'
        #print (formation)
#    for y in range (0, 3):
#        print (lines[x][y])


# In[41]:

strandArray['NucNum'] = dataArray
strandArray['Base'] = lstrand
strandArray['Tail'] = tArray
strandArray['Free'] = fArray
strandArray['Bulge'] = bArray
strandArray['Internal'] = iArray
strandArray['Multistem'] = msArray
strandArray['Helix3'] = h3Array
strandArray['Helix5'] = h5Array
strandArray['Hairpin'] = hpArray


# In[42]:

strandArray


# In[43]:

#Creation of Strand Array, manually copied from ALPSX files
eColiStrand = list('AAAUUGAAGAGUUUGAUCAUGGCUCAGAUUGAACGCUGGCGGCAGGCCUAACACAUGCAAGUCGAACGGUAACAGGAAGAAGCUUGCUUCUUUGCUGACGAGUGGCGGACGGGUGAGUAAUGUCUGGGAAACUGCCUGAUGGAGGGGGAUAACUACUGGAAACGGUAGCUAAUACCGCAUAACGUCGCAAGACCAAAGAGGGGGACCUUCGGGCCUCUUGCCAUCGGAUGUGCCCAGAUGGGAUUAGCUAGUAGGUGGGGUAACGGCUCACCUAGGCGACGAUCCCUAGCUGGUCUGAGAGGAUGACCAGCCACACUGGAACUGAGACACGGUCCAGACUCCUACGGGAGGCAGCAGUGGGGAAUAUUGCACAAUGGGCGCAAGCCUGAUGCAGCCAUGCCGCGUGUAUGAAGAAGGCCUUCGGGUUGUAAAGUACUUUCAGCGGGGAGGAAGGGAGUAAAGUUAAUACCUUUGCUCAUUGACGUUACCCGCAGAAGAAGCACCGGCUAACUCCGUGCCAGCAGCCGCGGUAAUACGGAGGGUGCAAGCGUUAAUCGGAAUUACUGGGCGUAAAGCGCACGCAGGCGGUUUGUUAAGUCAGAUGUGAAAUCCCCGGGCUCAACCUGGGAACUGCAUCUGAUACUGGCAAGCUUGAGUCUCGUAGAGGGGGGUAGAAUUCCAGGUGUAGCGGUGAAAUGCGUAGAGAUCUGGAGGAAUACCGGUGGCGAAGGCGGCCCCCUGGACGAAGACUGACGCUCAGGUGCGAAAGCGUGGGGAGCAAACAGGAUUAGAUACCCUGGUAGUCCACGCCGUAAACGAUGUCGACUUGGAGGUUGUGCCCUUGAGGCGUGGCUUCCGGAGCUAACGCGUUAAGUCGACCGCCUGGGGAGUACGGCCGCAAGGUUAAAACUCAAAUGAAUUGACGGGGGCCCGCACAAGCGGUGGAGCAUGUGGUUUAAUUCGAUGCAACGCGAAGAACCUUACCUGGUCUUGACAUCCACGGAAGUUUUCAGAGAUGAGAAUGUGCCUUCGGGAACCGUGAGACAGGUGCUGCAUGGCUGUCGUCAGCUCGUGUUGUGAAAUGUUGGGUUAAGUCCCGCAACGAGCGCAACCCUUAUCCUUUGUUGCCAGCGGUCCGGCCGGGAACUCAAAGGAGACUGCCAGUGAUAAACUGGAGGAAGGUGGGGAUGACGUCAAGUCAUCAUGGCCCUUACGACCAGGGCUACACACGUGCUACAAUGGCGCAUACAAAGAGAAGCGACCUCGCGAGAGCAAGCGGACCUCAUAAAGUGCGUCGUAGUCCGGAUUGGAGUCUGCAACUCGACUCCAUGAAGUCGGAAUCGCUAGUAAUCGUGGAUCAGAAUGCCACGGUGAAUACGUUCCCGGGCCUUGUACACACCGCCCGUCACACCAUGGGAGUGGGUUGCAAAAGAAGUAGGUAGCUUAACCUUCGGGAGGGCGCUUACCACUUUGUGAUUCAUGACUGGGGUGAAGUCGUAACAAGGUAACCGUAGGGGAACCUGCGGUUGGAUCACCUCCUUA')
mPneuStrand = list('NUUUUCUGAGAGUUUGAUCCUGGCUCAGGAUUAACGCUGGCGGCAUGCCUAAUACAUGCAAGUCGAUCGAAAGUAGUAAUACUUUAGAGGCGAACGGGUGAGUAACACGUAUCCAAUCUACCUUAUAAUGGGGGAUAACUAGUUGAAAGACUAGCUAAUACCGCAUAAGAACUUUGGUUCGCAUGAAUCAAAGUUGAAAGGACCUGCAAGGGUUCGUUAUUUGAUGAGGGUGCGCCAUAUCAGCUAGUUGGUGGGGUAACGGCCUACCAAGGCAAUGACGUGUAGCUAUGCUGAGAAGUAGAAUAGCCACAAUGGGACUGAGACACGGCCCAUACUCCUACGGGAGGCAGCAGUAGGGAAUUUUUCACAAUGAGCGAAAGCUUGAUGGAGCAAUGCCGCGUGAACGAUGAAGGUCUUUAAGAUUGUAAAGUUCUUUUAUUUGGGAAGAAUGACUUUAGCAGGUAAUGGCUAGAGUUUGACUGUACCAUUUUGAAUAAGUGACGACUAACUAUGUGCCAGCAGUCGCGGUAAUACAUAGGUCGCAAGCGUUAUCCGGAUUUAUUGGGCGUAAAGCAAGCGCAGGCGGAUUGAAAAGUCUGGUGUUAAAGGCAGCUGCUUAACAGUUGUAUGCAUUGGAAACUAUUAAUCUAGAGUGUGGUAGGGAGUUUUGGAAUUUCAUGUGGAGCGGUGAAAUGCGUAGAUAUAUGAAGGAACACCAGUGGCGAAGGCGAAAACUUAGGCCAUUACUGACGCUUAGGCUUGAAAGUGUGGGGAGCAAAUAGGAUUAGAUACCCUAGUAGUCCACACCGUAAACGAUAGAUACUAGCUGUCGGGGCGAUCCCCUCGGUAGUGAAGUUAACACAUUAAGUAUCUCGCCUGGGUAGUACAUUCGCAAGAAUGAAACUCAAACGGAAUUGACGGGGACCCGCACAAGUGGUGGAGCAUGUUGCUUAAUUCGACGGUACACGAAAAACCUUACCUAGACUUGACAUCCUUGGCAAAGUUAUGGAAACAUAAUGGAGGUUAACCGAGUGACAGGUGGUGCAUGGUUGUCGUCAGCUCGUGUCGUGAGAUGUUGGGUUAAGUCCCGCAACGAGCGCAACCCUUAUCGUUAGUUACAUUGUCUAGCGAGACUGCUAAUGCAAAUUGGAGGAAGGAAGGGAUGACGUCAAAUCAUCAUGCCCCUUAUGUCUAGGGCUGCAAACGUGCUACAAUGGCCAAUACAAACAGUCGCCAGCUUGUAAAAGUGAGCAAAUCUGUAAAGUUGGUCUCAGUUCGGAUUGAGGGCUGCAAUUCGUCCUCAUGAAGUCGGAAUCACUAGUAAUCGCGAAUCAGCUAUGUCGCGGUGAAUACGUUCUCGGGUCUUGUACACACCGCCCGUCAAACUAUGAAAGCUGGUAAUAUUUAAAAACGUGUUGCUAACCAUUAGGAAGCGCAUGUCAAGGAUAGCACCGGUGAUUGGAGUUAAGUCGUAACAAGGUACCCCUACGAGAACGUGGGGGUGGAUCACCUCCUUU')
bBurgStrand = list('AAAUAACGAAGAGUUUGAUCCUGGCUUAGAACUAACGCUGGCAGUGCGUCUUAAGCAUGCAAGUCAAACGGGAUGUAGCAAUACAUUCAGUGGCGAACGGGUGAGUAACGCGUGGAUGAUCUACCUAUGAGAUGGGGAUAACUAUUAGAAAUAGUAGCUAAUACCGAAUAAGGUCAGUUAAUUUGUUAAUUGAUGAAAGGAAGCCUUUAAAGCUUCGCUUGUAGAUGAGUCUGCGUCUUAUUAGCUAGUUGGUAGGGUAAAUGCCUACCAAGGCAAUGAUAAGUAACCGGCCUGAGAGGGUGAACGGUCACACUGGAACUGAGAUACGGUCCAGACUCCUACGGGAGGCAGCAGCUAAGAAUCUUCCGCAAUGGGCGAAAGCCUGACGGAGCGACACUGCGUGAAUGAAGAAGGUCGAAAGAUUGUAAAAUUCUUUUAUAAAUGAGGAAUAAGCUUUGUAGGAAAUGACAAAGUGAUGACGUUAAUUUAUGAAUAAGCCCCGGCUAAUUACGUGCCAGCAGCCGCGGUAAUACGUAAGGGGCGAGCGUUGUUCGGGAUUAUUGGGCGUAAAGGGUGAGUAGGCGGAUAUAUAAGUCUAUGCAUAAAAUACCACAGCUCAACUGUGGACCUAUGUUGGAAACUAUAUGUCUAGAGUCUGAUAGAGGAAGUUAGAAUUUCUGGUGUAAGGGUGGAAUCUGUUGAUAUCAGAAAGAAUACCGGAGGCGAAGGCGAACUUCUGGGUCAAGACUGACGCUGAGUCACGAAAGCGUAGGGAGCAAACAGGAUUAGAUACCCUGGUAGUCUACGCUGUAAACGAUGCACACUUGGUGUUAACUAAAAGUUAGUACCGAAGCUAACGUGUUAAGUGUGCCGCCUGGGGAGUAUGCUCGCAAGAGUGAAACUCAAAGGAAUUGACGGGGGCCCGCACAAGCGGUGGAGCAUGUGGUUUAAUUCGAUGAUACGCGAGGAACCUUACCAGGGCUUGACAUAUAUAGGAUAUAGUUAGAGAUAAUUAUUCCCCGUUUGGGGUCUAUAUACAGGUGCUGCAUGGUUGUCGUCAGCUCGUGCUGUGAGGUGUUGGGUUAAGUCCCGCAACGAGCGCAACCCUUGUUAUCUGUUACCAGCAUGUAAUGGUGGGGACUCAGAUAAGACUGCCGGUGAUAAGUCGGAGGAAGGUGAGGAUGACGUCAAAUCAUCAUGGCCCUUAUGUCCUGGGCUACACACGUGCUACAAUGGCCUGUACAAAGCGAAGCGAAACAGUGAUGUGAAGCAAAACGCAUAAAGCAGGUCUCAGUCCGGAUUGAAGUCUGAAACUCGACUUCAUGAAGUUGGAAUCGCUAGUAAUCGUAUAUCAGAAUGAUACGGUGAAUACGUUCUCGGGCCUUGUACACACCGCCCGUCACACCACCCGAGUUGAGGAUACCCGAAGCUAUUAUUCUAACCCGUAAGGGAGGAAGGUAUUUAAGGUAUGUUUAGUGAGGGGGGUGAAGUCGUAACAAGGUAGCCGUACUGGAAAGUGCGGCUGGAUCACCUCCUUU')
bHaloStrand = list('NUUUAUGGAGAGUUUGAUCCUGGCUCAGGACGAACGCUGGCGGCGUGCCUAAUACAUGCAAGUCGAGCGGACCAAAGGGAGCUUGCUCCUAGAGGUUAGCGGCGAACGGGUGAGUAACACGUGGGCAACCUGCCUGUAAGACUGGGAUAACAUCGAGAAAUCGGUGCUAAUACUGGAUAAUAAAAAGAACUGCAUGGUUCUUUUUUGAAAGAUGGUUUCGGCUAUCACUUACAGAUGGGCCCGCGGCGCAUUAGCUAGUUGGUGGGGUAACGGCUCACCAAGGCGACGAUGCGUAGCCGACCUGAGAGGGUGAUCGGCCACACUGGGACUGAGACACGGCCCAGACUCCUACGGGAGGCAGCAGUAGGGAAUCUUCCGCAAUGGACGAAAGUCUGACGGAGCAACGCCGCGUGAGUGAUGAAGGUUUUCGGAUCGUAAAACUCUGUUGUUAGGGAAGAACAAGUGCCGUUCGAAAGGGCGGCACCUUGACGGUACCUAACGAGAAAGCCACGGCUAACUACGUGCCAGCAGCCGCGGUAAUACGUAGGUGGCAAGCGUUGUCCGGAAUUAUUGGGCGUAAAGCGCGCGCAGGCGGUCUCUUAAGUCUGAUGUGAAAGCCCCCGGCUCAACCGGGGAGGGUCAUUGGAAACUGGGAGACUUGAGUACAGAAGAGGAGAGUGGAAUUCCACGUGUAGCGGUGAAAUGCGUAGAGAUGUGGAGGAACACCAGUGGCGAAGGCGACUCUCUGGUCUGUAACUGACGCUGAGGCGCGAAAGCGUGGGGAGCAAACAGGAUUAGAUACCCUGGUAGUCCACGCCGUAAACGAUGAGUGCUAGGUGUUAGGGGUUUCGACGCCCUUAGUGCCGAAGUUAACACAUUAAGCACUCCGCCUGGGGAGUACGACCGCAAGGUUGAAACUCAAAGGAAUUGACGGGGGCCCGCACAAGCAGUGGAGCAUGUGGUUUAAUUCGAAGCAACGCGAAGAACCUUACCAGGUCUUGACAUCCUUUGACCACCCUAGAGAUAGGGCUUUCCCCUUCGGGGGACAAAGUGACAGGUGGUGCAUGGUUGUCGUCAGCUCGUGUCGUGAGAUGUUGGGUUAAGUCCCGCAACGAGCGCAACCCUUGACCUUAGUUGCCAGCAUUCAGUUGGGCACUCUAAGGUGACUGCCGGUGACAAACCGGAGGAAGGUGGGGAUGACGUCAAAUCAUCAUGCCCCUUAUGACCUGGGCUACACACGUGCUACAAUGGAUGGUACAAAGGGUUGCGAAGCCGCGAGGUGAAGCCAAUCCCAGAAAGCCAUUCUCAGUUCGGAUUGCAGGCUGCAACUCGCCUGCAUGAAGCCGGAAUUGCUAGUAAUCGCGGAUCAGCAUGCCGCGGUGAAUACGUUCCCGGGCCUUGUACACACCGCCCGUCACACCACGAGAGUUUGUAACACCCGAAGUCGGUGGGGUAACCUUUUUGGAGCCAGCCGCCUAAGGUGGGACAGAUGAUUGGGGUGAAGUCGUAACAAGGUAGCCGUAUCGGAAGGUGCGGCUGGAUCACCUCCUUUCU')
pMultStrand = list('GAAUUGAAGAGUUUGAUCAUGGCUCAGAUUGAACGCUGGCGGCAGGCUUAACACAUGCAAGUCGAACGGUAGCAGGAAGAAAGCUUGCUUUCUUUGCUGACGAGUGGCGGACGGGUGAGUAAUGCUUGGGAAUCUGGCUUAUGGAGGGGGAUAACUGUGGGAAACUGCAGCUAAUACCGCGUAUUCUCUGAGGAGGAAAGGGUGGGACCUUAGGGCCACCUGCCAUAAGAUGAGCCCAAGUGGGAUUAGGUAGUUGGUGGGGUAAAGGCCUACCAAGCCUGCGAUCUCUAGCUGGUCUGAGAGGAUGACCAGCCACACUGGAACUGAGACACGGUCCAGACUCCUACGGGAGGCAGCAGUGGGGAAUAUUGCGCAAUGGGGGGAACCCUGACGCAGCCAUGCCGCGUGAAUGAAGAAGGCCUUCGGGUUGUAAAGUUCUUUCGGUAAUGAGGAAGGGAUGUUGUUAAAUAGAUAGCAUCAUUGACGUUAAUUACAGAAGAAGCACCGGCUAACUCCGUGCCAGCAGCCGCGGUAAUACGGAGGGUGCGAGCGUUAAUCGGAAUAACUGGGCGUAAAGGGCACGCAGGCGGACUUUUAAGUGAGAUGUGAAAUCCCCGAGCUUAACUUGGGAACUGCAUUUCAGACUGGGAGUCUAGAGUACUUUAGGGAGGGGUAGAAUUCCACGUGUAGCGGUGAAAUGCGUAGAGAUGUGGAGGAAUACCGAAGGCGAAGGCAGCCCCUUGGGAAUGUACUGACGCUCAUGUGCGAAAGCGUGGGGAGCAAACAGGAUUAGAUACCCUGGUAGUCCACGCUGUAAACGCUGUCGAUUUGGGGAUUGGGCUAUAUGCUUGGUGCCCGAAGCUAACGUGAUAAAUCGACCGCCUGGGGAGUACGGCCGCAAGGUUAAAACUCAAAUGAAUUGACGGGGGCCCGCACAAGCGGUGGAGCAUGUGGUUUAAUUCGAUGCAACGCGAAGAACCUUACCUACUCUUGACAUCCUAAGAAGAGCUCAGAGAUGAGCUUGUGCCUUCGGGAACUUAGAGACAGGUGCUGCAUGGCUGUCGUCAGCUCGUGUUGUGAAAUGUUGGGUUAAGUCCCGCAACGAGCGCAACCCUUAUCCUUUGUUGCCAGCGAUUCGGUCGGGAACUCAAAGGAGACUGCCAGUGACAAACUGGAGGAAGGUGGGGAUGACGUCAAGUCAUCAUGGUCCUUACGAGUAGGGCUACACACGUGCUACAAUGGUGCAUACAGAGGGCAGCGAGAGUGCGAGCUUGAGCGAAUCUCAGAAAGUGCAUCUAAGUCCGGAUUGGAGUCUGCAACUCGACUCCAUGAAGUCGGAAUCGCUAGUAAUCGCAAAUCAGAAUGUUGCGGUGAAUACGUUCCCGGGCCUUGUACACACCGCCCGUCACACCAUGGGAGUGGGUUGUACCAGAAGUAGAUAGCUUAACCUUCGGGAGGGCGUUUACCACGGUAUGAUUCAUGACUGGGGUGAAGUCGUAACAAGGUAACCGUAGGGGAACCUGCGGUUGGAUCACCUCCUUA')
rProwStrand = list('AUCAAACUUGAGAGUUUGAUCCUGGCUCAGAACGAACGCUAUCGGUAUGCUUAACACAUGCAAGUCGAACGGAUUAACUAGAGCUCGCUUUAGUUAAUUAGUGGCAGACGGGUGAGUAACACGUGGGAAUCUACCCAUCAGUACGGAAUAACUUUUAGAAAUAAAAGCUAAUACCGUAUAUUCUCUACGGAGGAAAGAUUUAUCGCUGAUGGAUGGGCCCGCGUCAGAUUAGGUAGUUGGUGAGGUAAUGGCUCACCAAGCCGACGAUCUGUAGCUGGUCUGAGAGGAUGAUCAGCCACACUGGGACUGAGACACGGCCCAGACUCCUACGGGAGGCAGCAGUGGGGAAUAUUGGACAAUGGGCGAAAGCCUGAUCCAGCAAUACCGAGUGAGUGAUGAAGGCCUUAGGGUUGUAAAGCUCUUUUAGCAAGGAAGAUAAUGACGUUACUUGCAGAAAAAGCCCCGGCUAACUCCGUGCCAGCAGCCGCGGUAAGACGGAGGGGGCUAGCGUUGUUCGGAAUUACUGGGCGUAAAGAGUGCGUAGGCGGUUUAGUAAGUUGGAAGUGAAAGCCCGGGGCUUAACCUCGGAAUUGCUUUCAAAACUACUAAUCUAGAGUGUAGUAGGGGAUGAUGGAAUUCCUAGUGUAGAGGUGAAAUUCUUAGAUAUUAGGAGGAACACCGGUGGCGAAGGCGGUCAUCUGGGCUACAACUGACGCUGAUGCACGAAAGCGUGGGGAGCAAACAGGAUUAGAUACCCUGGUAGUCCACGCCGUAAACGAUGAGUGCUAGAUAUCGGAGGAUUCUCUUUCGGUUUCGCAGCUAACGCAUUAAGCACUCCGCCUGGGGAGUACGGUCGCAAGAUUAAAACUCAAAGGAAUUGACGGGGGCUCGCACAAGCGGUGGAGCAUGCGGUUUAAUUCGAUGUUACGCGAAAAACCUUACCAACCCUUGACAUGGUGGUUACGGAUUGCAGAGAUGCUUUCCUUCAGUUCGGCUGGGCCACACACAGGUGUUGCAUGGCUGUCGUCAGCUCGUGUCGUGAGAUGUUGGGUUAAGUCCCGCAACGAGCGCAACCCUUAUUCUUAUUUGCCAGUGGGUAAUGCCGGGAACUAUAAGAAAACUGCCGGUGAUAAGCCGGAGGAAGGUGGGGACGACGUCAAGUCAUCAUGGCCCUUACGGGUUGGGCUACACGCGUGCUACAAUGGUGUUUACAGAGGGAAGCAAUACGGUGACGUGGAGCAAAUCCCUAAAAGACAUCUCAGUUCGGAUUGUUCUCUGCAACUCGAGAGCAUGAAGUUGGAAUCGCUAGUAAUCGCGGAUCAGCAUGCCGCGGUGAAUACGUUCUCGGGCCUUGUACACACUGCCCGUCACGCCAUGGGAGUUGGUUUUACCUGAAGGUGGUGAGCUAACGCAAGAGGCAGCCAACCACGGUAAAAUUAGCGACUGGGGUGAAGUCGUAACAAGGUAGCCGUAGGGGAACUGCGGCUGGAUUACCUCCUUA')
tMariStrand = list('UAUAUGGAGGGUUUGAUCCUGGCUCAGGGUGAACGCUGGCGGCGUGCCUAACACAUGCAAGUCGAGCGGGGGAAACUCCCUUCGGGGAGGAGUACCCAGCGGCGGACGGGUGAGUAACACGUGGGUAACCUGCCCUCCGGAGGGGGAUAACCAGGGGAAACCCUGGUUAAUACCCCAUACGCUCCAUCAACGCAAGUUGGUGGAGGAAAGGGGCGUUUGCCCCGCCGGAGGAGGGGCCCGCGGCCCAUCAGGUAGUUGGUGGGGUAACGGCCCACCAAGCCGACGACGGGUAGCCGGCCUGAGAGGGUGGUCGGCCACAGGGGCACUGAGACACGGGCCCCACUCUACGGGAGGCAGCAGUGGGGAAUCUUGGACAAUGGGGGAAACCCUGAUCCAGCGACGCCGCGUGCGGGACGAAGCCUUCGGGGUGUAAACCGCUGUGGCGGGGGAAGAAUAAGGUAGGGAGGAAAUGCCCUACCGAUGACGGUACCCCGCUAGAAAGCCCCGGCUAACUACGUGCCAGCAGCCGCGGUAAUACGUAGGGGGCAAGCGUUACCCGGAUUUACUGGGCGUAAAGGGGGCGUAGGCGGCCUGGUGUGUCGGAUGUGAAAUCCCACGGCUCAACCGUGGGGCUGCAUCCGAAACUACCAGGCUUGGGGGCGGUAGAGGGAGACGGAACUGCCGGUGUAGGGGUGAAAUCCGUAGAUAUCGGCAGGAACGCCGGUGGGGAAGCCGGUCUCCUGGGCCGACCCCGACGCUGAGGCCCGAAAGCCAGGGGAGCAAACCGGAUUAGAUACCCGGGUAGUCCUGGCUGUAAACGAUGCCCACUAGGUGUGGGGGGGUAAUCCCUCCGUGCUGAAGCUAACGCGUUAAGUGGGCCGCCUGGGGAGUACGCCCGCAAGGGUGAAACUCAAAGGAAUUGACGGGGGCCCGCACAAGCGGUGGAGCGUGUGGUUUAAUUGGAUGCUAAGCCAAGAACCUUACCAGGGCUUGACAUGCCGGUGGUACCUCCCCGAAAGGGGUAGGGACCCAGUCCUUCGGGACUGGGAGCCGGCACAGGUGGUGCACGGCCGUCGUCAGCUCGUGCCGUGAGGUGUUGGGUUAAGUCCCGCAACGAGCGCAACCCCUGCCCCUAGUUGCCAGCGGUUCGGCCGGGCACUCUAGGGGGACUGCCGGCGACGAGCCGGAGGAAGGAGGGGAUGACGUCAGGUACUCGUGCCCCUUAUGCCCUGGGCGACACACGCGCUACAAUGGGCGGUACAAUGGGUUGCGACCCCGCGAGGGGGAGCCAAUCCCCAAAGCCGCCCUCAGUUCGGAUCGCAGGCUGCAACCCGCCUGCGUGAAGCCGGAAUCGCUAGUAAUCGCGGAUCAGCCAUGCCGCGGUGAAUACGUUCCCGGGCCUUGUACACACCGCCCGUCACGCCACCCGAGUCGGGGGCUCCCGAAGACACCUACCCCAACCCGAAAGGGAGGGGGGGUGUCGAGGGAGAACCUGGCGAGGGGGGCGAAGUCGUAACAAGGUAGCCGUACCGGAAGGUGCGGCUGGAUCACCUCCUUUC')


# In[44]:

#Test code of concatenating list into single column
eCStrandA = pd.DataFrame(eColiStrand)
eCStrandA = pd.concat([eCStrandA.T[x] for x in eCStrandA.T], ignore_index='true')


# In[45]:

#Definition for above functions
def listToCol(strand):
    df = pd.DataFrame(strand)
    out = pd.concat([df.T[i] for i in df.T], ignore_index='true')
    return out


# In[46]:

#New column variables set with nucleotides of organism's RNA
#These values are used in definitions later on
eCCol = listToCol(eColiStrand)
mPCol = listToCol(mPneuStrand)
bBCol = listToCol(bBurgStrand)
bHCol = listToCol(bHaloStrand)
pMCol = listToCol(pMultStrand)
rPCol = listToCol(rProwStrand)
tMCol = listToCol(tMariStrand)


# In[47]:

def createTable(filename):
    dataArray = []
    b5Array = []
    b3Array = []
    hpArray = []
    hkArray = []
    h5Array = []
    h3Array = []
    msArray = []
    i5Array = []
    i3Array = []
    fArray = []
    tArray = []
    strandArray = pd.DataFrame(columns=['NucNum','Base','Hairpin','Helix5','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5','Bulge3','Free','Tail'])
    if filename == "E Coli Stripped.txt":
        for x in range (0, len(eCCol)):
            dataArray.append(x + 1)
            tArray.append('FALSE')
            fArray.append('FALSE')
            b5Array.append('FALSE')
            b3Array.append('FALSE')
            i5Array.append('FALSE')
            i3Array.append('FALSE')
            msArray.append('FALSE')
            h3Array.append('FALSE')
            h5Array.append('FALSE')
            hkArray.append('FALSE')
            hpArray.append('FALSE')
        strandArray['Base'] = eCCol
    elif filename == "M Pneumoniae Stripped.txt":
        for x in range (1, len(mPCol) + 1):
            dataArray.append(x)
            tArray.append('FALSE')
            fArray.append('FALSE')
            b5Array.append('FALSE')
            b3Array.append('FALSE')
            i5Array.append('FALSE')
            i3Array.append('FALSE')
            msArray.append('FALSE')
            h3Array.append('FALSE')
            h5Array.append('FALSE')
            hkArray.append('FALSE')
            hpArray.append('FALSE')
        strandArray['Base'] = mPCol
    elif filename == "B Burgdorferi Stripped.txt":
        for x in range (1, len(bBCol) + 1):
            dataArray.append(x)
            tArray.append('FALSE')
            fArray.append('FALSE')
            b5Array.append('FALSE')
            b3Array.append('FALSE')
            i5Array.append('FALSE')
            i3Array.append('FALSE')
            msArray.append('FALSE')
            h3Array.append('FALSE')
            h5Array.append('FALSE')
            hkArray.append('FALSE')
            hpArray.append('FALSE')
        strandArray['Base'] = bBCol
    elif filename == "B Halodurans Stripped.txt":
        for x in range (1, len(bHCol) + 1):
            dataArray.append(x)
            tArray.append('FALSE')
            fArray.append('FALSE')
            b5Array.append('FALSE')
            b3Array.append('FALSE')
            i5Array.append('FALSE')
            i3Array.append('FALSE')
            msArray.append('FALSE')
            h3Array.append('FALSE')
            h5Array.append('FALSE')
            hkArray.append('FALSE')
            hpArray.append('FALSE')
        strandArray['Base'] = bHCol
    elif filename == "P Multocida Stripped.txt":
        for x in range (1, len(pMCol) + 1):
            dataArray.append(x)
            tArray.append('FALSE')
            fArray.append('FALSE')
            b5Array.append('FALSE')
            b3Array.append('FALSE')
            i5Array.append('FALSE')
            i3Array.append('FALSE')
            msArray.append('FALSE')
            h3Array.append('FALSE')
            h5Array.append('FALSE')
            hkArray.append('FALSE')
            hpArray.append('FALSE')
        strandArray['Base'] = pMCol
    elif filename == "R Prowazekii Stripped.txt":
        for x in range (1, len(rPCol) + 1):
            dataArray.append(x)
            tArray.append('FALSE')
            fArray.append('FALSE')
            b5Array.append('FALSE')
            b3Array.append('FALSE')
            i5Array.append('FALSE')
            i3Array.append('FALSE')
            msArray.append('FALSE')
            h3Array.append('FALSE')
            h5Array.append('FALSE')
            hkArray.append('FALSE')
            hpArray.append('FALSE')
        strandArray['Base'] = rPCol
    elif filename == "T Maritima Stripped.txt":
        for x in range (1, len(tMCol) + 1):
            dataArray.append(x)
            tArray.append('FALSE')
            fArray.append('FALSE')
            b5Array.append('FALSE')
            b3Array.append('FALSE')
            i5Array.append('FALSE')
            i3Array.append('FALSE')
            msArray.append('FALSE')
            h3Array.append('FALSE')
            h5Array.append('FALSE')
            hkArray.append('FALSE')
            hpArray.append('FALSE')
        strandArray['Base'] = tMCol
    with open(filename) as textFile:
        lines = [line.split() for line in textFile]
    for x in range (0, len(lines)):
        lower = lines[x][0]
        lower = int(lower) - 1
        upper = lines[x][1]
        upper = int(upper)
        formation = lines[x][2]
        if formation == 'TAIL':
            for i in range (lower, upper):
                tArray[i] = 'TRUE'
        elif formation == 'FREE':
            for i in range (lower, upper):
                fArray[i] = 'TRUE'
        elif formation == 'BULGE5':
            for i in range (lower, upper):
                b5Array[i] = 'TRUE'
        elif formation == 'BULGE3':
            for i in range (lower, upper):
                b3Array[i] = 'TRUE'
        elif formation == 'INTERNAL5':
            for i in range (lower, upper):
                i5Array[i] = 'TRUE'
        elif formation == 'INTERNAL3':
            for i in range (lower, upper):
                i3Array[i] = 'TRUE'
        elif formation == 'MULTISTEM':
            for i in range (lower, upper):
                msArray[i] = 'TRUE'
        elif formation == 'HELIX3':
            for i in range (lower, upper):
                h3Array[i] = 'TRUE'
        elif formation == 'HELIX5':
            for i in range (lower, upper):
                h5Array[i] = 'TRUE'
        elif formation == 'HELIXKNOT':
            for i in range (lower, upper):
                hkArray[i] = 'TRUE'
        elif formation == 'HAIRPIN':
            for i in range (lower, upper):
                hpArray[i] = 'TRUE'
    strandArray['NucNum'] = dataArray
    strandArray['Tail'] = tArray
    strandArray['Free'] = fArray
    strandArray['Bulge5'] = b5Array
    strandArray['Bulge3'] = b3Array
    strandArray['Internal5'] = i5Array
    strandArray['Internal3'] = i3Array
    strandArray['Multistem'] = msArray
    strandArray['Helix3'] = h3Array
    strandArray['Helix5'] = h5Array
    strandArray['HelixKnot'] = hkArray
    strandArray['Hairpin'] = hpArray
    return strandArray


# In[48]:

eCTable = createTable("T Maritima Stripped.txt")
mPTable = createTable("M Pneumoniae Stripped.txt")
bBTable = createTable("B Burgdorferi Stripped.txt")
bHTable = createTable("B Halodurans Stripped.txt")
pMTable = createTable("P Multocida Stripped.txt")
rPTable = createTable("R Prowazekii Stripped.txt")
tMTable = createTable("T Maritima Stripped.txt")


# In[49]:

#Create giant test set without overwritten data
#Hold out E Coli and M Pneumoniae as test sets
trainSet = pd.DataFrame(columns=['NucNum','Base','Hairpin','Helix5','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5','Bulge3','Free','Tail'])
trainSet = bBTable
trainSet = trainSet.append(bHTable)
trainSet = trainSet.append(pMTable)
trainSet = trainSet.append(rPTable)
trainSet = trainSet.append(tMTable)


# In[50]:

#Create copy of test set with booleans swapped for integers
d = {'TRUE':1, 'FALSE':0}
trainNotBool = trainSet.applymap(lambda x: d.get(x,x))
trainNotBool
eCTestNB = eCTable.applymap(lambda x: d.get(x,x))
mPTestNB = mPTable.applymap(lambda x: d.get(x,x))


# In[76]:

trainSet


# In[98]:

#Start Data Collection
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
#testSet.Tail.value_counts().plot(kind='bar')
g = sns.stripplot(x="HelixKnot", y="NucNum", data=trainSet, jitter=True);
g.axes.set_title('Helix-Knot at a Given Nucleotide Position', fontsize=20,alpha=0.75)
#tips = sns.load_dataset("tips")
#g = sns.FacetGrid(data=trainSet, col="NucNum", row="Tail", margin_titles=True)
#g.map(sns.plt.scatter, "NucNum", "Tail")


# In[100]:

#Start training data model
#Use classification techniques
#lrTest = testSet.columns[0:2]
#var_mod = ['Base']
#le = LabelEncoder()
#for i in var_mod:
#    lrTest[i] = le.fit_transform(lrTest[i].factorize()[0])
#model = LogisticRegression()
#model = model.fit(lrTest, testSet['Hairpin'])
#model.score(lrTest, testSet['Hairpin'])
import statsmodels.api as sm
dummy_ranks = pd.get_dummies(trainNotBool['Base'], prefix='Base')
#dummy_ranksEC = pd.get_dummies(eCTestNB['Base'], prefix='Base')
#dummy_ranksMP = pd.get_dummies(mPTestNB['Base'], prefix='Base')
print (dummy_ranks.head())


# In[101]:

trainKeep = ['NucNum', 'Hairpin', 'Helix5','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5','Bulge3','Free','Tail']
trainData = trainNotBool[trainKeep].join(dummy_ranks.ix[:, 'Base_A':])
testDataEC = eCTestNB[trainKeep].join(dummy_ranks.ix[:, 'Base_A':])
testDataMP = mPTestNB[trainKeep].join(dummy_ranks.ix[:, 'Base_A':])
print (trainData.head())
#trainData['intercept'] = 1.0
#testDataEC['intercept'] = 1.0
#testDataMP['intercept'] = 1.0


# In[170]:

from sklearn.preprocessing import MinMaxScaler
train_cols = ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U','Helix5','Helix3','HelixKnot']
#'Helix5','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5','Bulge3','Free','Tail']
#train_colss = trainData[train_cols]
#train_cols = testData1.columns[3:]
#train_cols.astype(float)
# Index([gre, gpa, prestige_2, prestige_3, prestige_4], dtype=object)
##Logit runs until Bulge
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(trainData), columns=trainData.columns)
logit = sm.Logit(df_scaled['Hairpin'], df_scaled[train_cols])

# fit the model
result = logit.fit()
print (result.summary())
#print (df_scaled[train_cols])
#np.asarray(testData1['Hairpin']).dtype
#np.asarray(train_colss).dtype


# In[169]:

model = LogisticRegression()
model = model.fit(df_scaled[train_cols], df_scaled['Hairpin'])
print (model.score(df_scaled[train_cols], df_scaled['Hairpin']))
print (model.coef_)


# In[147]:

predict = model.predict(testDataEC[train_cols])
#len(predict)
#len(testDataEC['Hairpin'])
tdECHP = testDataEC['Hairpin']
for i in range(0, len(predict)):
    if predict[i] == 1:
        print ('TRUE')
    #else:
    #    print ("DONE")
#for j in range(0, len(predict)):
#    print(predict[j])


# In[134]:

tdECHP[0]


# In[113]:

#eCResult = logit.predict(testDataMP[train_cols])
#testDataEC[train_cols]


# In[104]:

from sklearn import svm
X = trainData[train_cols]
y = trainData['Hairpin']
clf = svm.SVC()
clf.fit(X, y)  


# In[60]:

pred = clf.predict(testDataEC[train_cols])


# In[61]:

compArray = pd.DataFrame(columns=['Predict','True'])
result = []
resetList = testDataEC.reset_index()
compArray['Predict'] = pred
compArray['True'] = resetList['Hairpin']
for i in compArray.index:
    #print (compArray.iloc[i][0]==1 and compArray.iloc[i][1]==1)
    result.append(compArray.iloc[i][0]==compArray.iloc[i][1])
len(result)
    #print (i, 'hi')
#for i in range(0, len(pred)):
#    print (pred[i])
#print(testDataEC['Hairpin'])
#set(pred).intersection(testDataEC['Hairpin'])
#[i for i, item in enumerate(pred) if item in testDataEC['Hairpin']]


# In[62]:

dataCollection = pd.DataFrame(columns=['VariableCount','Score'])
varCount = [] 
scoreSet = []
varCount[9]=11
#scoreSet.append(np.sum(result)/len(result))
print (np.sum(result)/len(result))


# In[853]:

dataCollection['VariableCount'] = varCount
dataCollection['Score'] = scoreSet
dataCollection
plt.plot(dataCollection['VariableCount'], dataCollection['Score'], '-o')
plt.title('Ratio of Correct Predictions for Hairpin Loop Formation')
plt.show()


# In[105]:

def testHairpin(filename):
    trainKeep = ['NucNum', 'Hairpin', 'Helix5','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5','Bulge3','Free','Tail']
    trainData = trainNotBool[trainKeep].join(dummy_ranks.ix[:, 'Base_A':])
    testDataEC = eCTestNB[trainKeep].join(dummy_ranks.ix[:, 'Base_A':])
    testDataMP = mPTestNB[trainKeep].join(dummy_ranks.ix[:, 'Base_A':])
    train_cols = ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Helix5','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5','Bulge3','Free','Tail']
    testCases=[['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Helix5'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Helix5','Helix3'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Helix5','Helix3','HelixKnot'],
              ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Helix5','Helix3','HelixKnot','Multistem'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Helix5','Helix3','HelixKnot','Multistem','Internal5'],
              ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Helix5','Helix3','HelixKnot','Multistem','Internal5','Internal3'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Helix5','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5'],
              ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Helix5','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5','Bulge3'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Helix5','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5','Bulge3','Free'],
              ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Helix5','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5','Bulge3','Free','Tail']]
    dataCollection = pd.DataFrame(columns=['VariableCount','Score'])
    varCount = [] 
    scoreSet = []
    if filename == 'ECOLI':
        for i in range(0, len(testCases)):
            X = trainData[testCases[i]]
            y = trainData['Hairpin']
            clf = svm.SVC()
            clf.fit(X, y)  
            pred = clf.predict(testDataEC[testCases[i]])
            compArray = pd.DataFrame(columns=['Predict','True'])
            result = []
            resetList = testDataEC.reset_index()
            compArray['Predict'] = pred
            compArray['True'] = resetList['Hairpin']
            for n in compArray.index:
                #print (compArray.iloc[n][0]==1 and compArray.iloc[n][1]==1)
                result.append(compArray.iloc[n][0]==compArray.iloc[n][1])
            varCount.append(i+2)
            scoreSet.append(np.sum(result)/len(result))
        dataCollection['VariableCount'] = varCount
        dataCollection['Score'] = scoreSet
        return dataCollection
#        plt.plot(dataCollection['VariableCount'], dataCollection['Score'], '-o')
#        plt.axis([1, 13, 0.8, 1])
#        plt.title('Ratio of Correct Predictions for Hairpin Loop Formation')
#        plt.show()
    if filename == 'MPNEUM':
        for i in range(0, len(testCases)):
            X = trainData[testCases[i]]
            y = trainData['Hairpin']
            clf = svm.SVC()
            clf.fit(X, y)  
            pred = clf.predict(testDataMP[testCases[i]])
            compArray = pd.DataFrame(columns=['Predict','True'])
            result = []
            resetList = testDataMP.reset_index()
            compArray['Predict'] = pred
            compArray['True'] = resetList['Hairpin']
            for n in compArray.index:
                #print (compArray.iloc[n][0]==1 and compArray.iloc[n][1]==1)
                result.append(compArray.iloc[n][0]==compArray.iloc[n][1])
            varCount.append(i+2)
            scoreSet.append(np.sum(result)/len(result))
        dataCollection['VariableCount'] = varCount
        dataCollection['Score'] = scoreSet
        return dataCollection
#        plt.axis([1, 13, 0.8, 1])
#        plt.title('Ratio of Correct Predictions for Hairpin Loop Formation')
#        plt.plot(dataCollection['VariableCount'], dataCollection['Score'], '-o')
#        plt.show()


# In[106]:

def testHelix(filename):
    trainKeep = ['NucNum', 'Hairpin', 'Helix5','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5','Bulge3','Free','Tail']
    trainData = trainNotBool[trainKeep].join(dummy_ranks.ix[:, 'Base_A':])
    testDataEC = eCTestNB[trainKeep].join(dummy_ranks.ix[:, 'Base_A':])
    testDataMP = mPTestNB[trainKeep].join(dummy_ranks.ix[:, 'Base_A':])
    train_cols = ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Helix5','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5','Bulge3','Free','Tail']
    testCases=[['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U','Hairpin'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Hairpin','Helix3'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Hairpin','Helix3','HelixKnot'],
              ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Hairpin','Helix3','HelixKnot','Multistem'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Hairpin','Helix3','HelixKnot','Multistem','Internal5'],
              ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Hairpin','Helix3','HelixKnot','Multistem','Internal5','Internal3'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U','Hairpin','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5'],
              ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Hairpin','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5','Bulge3'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Hairpin','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5','Bulge3','Free'],
              ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Hairpin','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5','Bulge3','Free','Tail']]
    dataCollection = pd.DataFrame(columns=['VariableCount','Score'])
    varCount = [] 
    scoreSet = []
    if filename == 'ECOLI':
        for i in range(0, len(testCases)):
            X = trainData[testCases[i]]
            y = trainData['Helix5']
            clf = svm.SVC()
            clf.fit(X, y)  
            pred = clf.predict(testDataEC[testCases[i]])
            compArray = pd.DataFrame(columns=['Predict','True'])
            result = []
            resetList = testDataEC.reset_index()
            compArray['Predict'] = pred
            compArray['True'] = resetList['Helix5']
            for n in compArray.index:
                #print (compArray.iloc[n][0]==1 and compArray.iloc[n][1]==1)
                result.append(compArray.iloc[n][0]==compArray.iloc[n][1])
            varCount.append(i+2)
            scoreSet.append(np.sum(result)/len(result))
        dataCollection['VariableCount'] = varCount
        dataCollection['Score'] = scoreSet
        return dataCollection
#        plt.plot(dataCollection['VariableCount'], dataCollection['Score'], '-o')
#        plt.axis([1, 13, 0.8, 1])
#        plt.title('Ratio of Correct Predictions for Hairpin Loop Formation')
#        plt.show()
    if filename == 'MPNEUM':
        for i in range(0, len(testCases)):
            X = trainData[testCases[i]]
            y = trainData['Helix5']
            clf = svm.SVC()
            clf.fit(X, y)  
            pred = clf.predict(testDataMP[testCases[i]])
            compArray = pd.DataFrame(columns=['Predict','True'])
            result = []
            resetList = testDataMP.reset_index()
            compArray['Predict'] = pred
            compArray['True'] = resetList['Helix5']
            for n in compArray.index:
                #print (compArray.iloc[n][0]==1 and compArray.iloc[n][1]==1)
                result.append(compArray.iloc[n][0]==compArray.iloc[n][1])
            varCount.append(i+2)
            scoreSet.append(np.sum(result)/len(result))
        dataCollection['VariableCount'] = varCount
        dataCollection['Score'] = scoreSet
        return dataCollection
#        plt.axis([1, 13, 0.8, 1])
#        plt.title('Ratio of Correct Predictions for Hairpin Loop Formation')
#        plt.plot(dataCollection['VariableCount'], dataCollection['Score'], '-o')
#        plt.show()


# In[107]:

def testMultistem(filename):
    trainKeep = ['NucNum', 'Hairpin', 'Helix5','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5','Bulge3','Free','Tail']
    trainData = trainNotBool[trainKeep].join(dummy_ranks.ix[:, 'Base_A':])
    testDataEC = eCTestNB[trainKeep].join(dummy_ranks.ix[:, 'Base_A':])
    testDataMP = mPTestNB[trainKeep].join(dummy_ranks.ix[:, 'Base_A':])
    train_cols = ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Helix5','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5','Bulge3','Free','Tail']
    testCases=[['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U','Hairpin'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U','Hairpin','Helix5'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U','Hairpin','Helix5','Helix3'],
              ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U','Hairpin','Helix5','Helix3','HelixKnot'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U','Hairpin','Helix5','Helix3','HelixKnot','Internal5'],
              ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U','Hairpin','Helix5','Helix3','HelixKnot','Internal5','Internal3'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U','Hairpin','Helix5','Helix3','HelixKnot','Internal5','Internal3','Bulge5'],
              ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U','Hairpin','Helix5','Helix3','HelixKnot','Internal5','Internal3','Bulge5','Bulge3'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U','Hairpin','Helix5','Helix3','HelixKnot','Internal5','Internal3','Bulge5','Bulge3','Free'],
              ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U','Hairpin','Helix5','Helix3','HelixKnot','Internal5','Internal3','Bulge5','Bulge3','Free','Tail']]
    dataCollection = pd.DataFrame(columns=['VariableCount','Score'])
    varCount = [] 
    scoreSet = []
    if filename == 'ECOLI':
        for i in range(0, len(testCases)):
            X = trainData[testCases[i]]
            y = trainData['Multistem']
            clf = svm.SVC()
            clf.fit(X, y)  
            pred = clf.predict(testDataEC[testCases[i]])
            compArray = pd.DataFrame(columns=['Predict','True'])
            result = []
            resetList = testDataEC.reset_index()
            compArray['Predict'] = pred
            compArray['True'] = resetList['Multistem']
            for n in compArray.index:
                #print (compArray.iloc[n][0]==1 and compArray.iloc[n][1]==1)
                result.append(compArray.iloc[n][0]==compArray.iloc[n][1])
            varCount.append(i+2)
            scoreSet.append(np.sum(result)/len(result))
        dataCollection['VariableCount'] = varCount
        dataCollection['Score'] = scoreSet
        return dataCollection
#        plt.plot(dataCollection['VariableCount'], dataCollection['Score'], '-o')
#        plt.axis([1, 13, 0.8, 1])
#        plt.title('Ratio of Correct Predictions for Hairpin Loop Formation')
#        plt.show()
    if filename == 'MPNEUM':
        for i in range(0, len(testCases)):
            X = trainData[testCases[i]]
            y = trainData['Multistem']
            clf = svm.SVC()
            clf.fit(X, y)  
            pred = clf.predict(testDataMP[testCases[i]])
            compArray = pd.DataFrame(columns=['Predict','True'])
            result = []
            resetList = testDataMP.reset_index()
            compArray['Predict'] = pred
            compArray['True'] = resetList['Multistem']
            for n in compArray.index:
                #print (compArray.iloc[n][0]==1 and compArray.iloc[n][1]==1)
                result.append(compArray.iloc[n][0]==compArray.iloc[n][1])
            varCount.append(i+2)
            scoreSet.append(np.sum(result)/len(result))
        dataCollection['VariableCount'] = varCount
        dataCollection['Score'] = scoreSet
        return dataCollection
#        plt.axis([1, 13, 0.8, 1])
#        plt.title('Ratio of Correct Predictions for Hairpin Loop Formation')
#        plt.plot(dataCollection['VariableCount'], dataCollection['Score'], '-o')
#        plt.show()


# In[108]:

dataCollection = pd.DataFrame(columns=['VariableCount','Score'])
dataCollection['VariableCount'] = varCount
dataCollection['Score'] = scoreSet
plt.plot(dataCollection['VariableCount'], dataCollection['Score'], '-o')
plt.axis([1,13,.8,1])
plt.title('Ratio of Correct Predictions for Hairpin Loop Formation')
plt.show()


# In[68]:

eCPredHP = testHairpin('ECOLI')


# In[69]:

eCPredHx = testHelix('ECOLI')


# In[176]:

eCPredMS = testMultistem('ECOLI')


# In[174]:

plt.plot(eCPredHP['VariableCount'], eCPredHP['Score'], '-o', label='Hairpin')
plt.plot(eCPredHx['VariableCount'], eCPredHx['Score'], '-o', label='Helix5')
plt.plot(eCPredMS['VariableCount'], eCPredMS['Score'], '-o', label='Multistem')
plt.legend(loc='lower right')
plt.axis([1,13,.8,1])
plt.title('Ratio of Correct Predictions for Various RNA Folding Structures for E. coli')
plt.show()


# In[896]:

mPPredHP = testHairpin('MPNEUM')


# In[918]:

mPPredHx = testHelix('MPNEUM')


# In[177]:

mPPredMS = testMultistem('MPNEUM')


# In[971]:

plt.plot(mPPredHP['VariableCount'], mPPredHP['Score'], '-o', label='Hairpin')
plt.plot(mPPredHx['VariableCount'], mPPredHx['Score'], '-o', label='Helix5')
plt.plot(mPPredMS['VariableCount'], mPPredMS['Score'], '-o', label='Multistem')
plt.legend(loc='lower right')
plt.axis([1,13,.6,1.1])
plt.xlabel('')
plt.title('Ratio of Correct Predictions for Various RNA Folding Structures for M. pneumoniae')
plt.show() #Do a logit at taper off point


# In[180]:

plt.plot(eCPredHP['VariableCount'], eCPredHP['Score'], '-o', label='E. coli Hairpin')
plt.plot(eCPredHx['VariableCount'], eCPredHx['Score'], '-o', label='E. coli Helix5')
plt.plot(eCPredMS['VariableCount'], eCPredMS['Score'], '-o', label='E. coli Multistem')
plt.plot(mPPredHP['VariableCount'], mPPredHP['Score'], '-^', label='M. pneumoniae Hairpin')
plt.plot(mPPredHx['VariableCount'], mPPredHx['Score'], '-^', label='M. pneumoniae Helix5')
plt.plot(mPPredMS['VariableCount'], mPPredMS['Score'], '-^', label='M. pneumoniae Multistem')
plt.legend(loc='lower right')
plt.xlabel('Number of Variables')
plt.ylabel('Ratio Correct')
plt.axis([1,13,.6,1.1])
plt.title('Ratio of Correct Predictions for Various RNA Folding Structures')


# In[903]:

#Random Forest
from sklearn.ensemble import RandomForestClassifier

clfRF = RandomForestClassifier(n_jobs=4)
clfRF.fit(trainData[train_cols], trainData['Hairpin'])

preds = clfRF.predict(testDataEC[train_cols])
pd.crosstab(testDataEC['Hairpin'], preds, rownames=['actual'], colnames=['preds'])


# In[905]:

clfRF = RandomForestClassifier(n_jobs=4)
clfRF.fit(trainData[train_cols], trainData['Hairpin'])

preds = clfRF.predict(testDataEC[train_cols])
pd.crosstab(testDataEC['Hairpin'], preds, rownames=['actual'], colnames=['preds'])


# In[984]:

def testMultistemRF(filename):
    trainKeep = ['NucNum', 'Hairpin', 'Helix5','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5','Bulge3','Free','Tail']
    trainData = trainNotBool[trainKeep].join(dummy_ranks.ix[:, 'Base_A':])
    testDataEC = eCTestNB[trainKeep].join(dummy_ranks.ix[:, 'Base_A':])
    testDataMP = mPTestNB[trainKeep].join(dummy_ranks.ix[:, 'Base_A':])
    train_cols = ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Helix5','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5','Bulge3','Free','Tail']
    testCases=[['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U','Hairpin'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U','Hairpin','Helix5'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U','Hairpin','Helix5','Helix3'],
              ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U','Hairpin','Helix5','Helix3','HelixKnot'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U','Hairpin','Helix5','Helix3','HelixKnot','Internal5'],
              ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U','Hairpin','Helix5','Helix3','HelixKnot','Internal5','Internal3'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U','Hairpin','Helix5','Helix3','HelixKnot','Internal5','Internal3','Bulge5'],
              ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U','Hairpin','Helix5','Helix3','HelixKnot','Internal5','Internal3','Bulge5','Bulge3'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U','Hairpin','Helix5','Helix3','HelixKnot','Internal5','Internal3','Bulge5','Bulge3','Free'],
              ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U','Hairpin','Helix5','Helix3','HelixKnot','Internal5','Internal3','Bulge5','Bulge3','Free','Tail']]
    dataCollection = pd.DataFrame(columns=['VariableCount','Score'])
    varCount = [] 
    scoreSet = []
    if filename == 'ECOLI':
        for i in range(0, len(testCases)):
            X = trainData[testCases[i]]
            y = trainData['Multistem']
            clf = RandomForestClassifier(n_jobs=4)
            clf.fit(X, y)  
            pred = clf.predict(testDataEC[testCases[i]])
            compArray = pd.DataFrame(columns=['Predict','True'])
            result = []
            resetList = testDataEC.reset_index()
            compArray['Predict'] = pred
            compArray['True'] = resetList['Multistem']
            print (pd.crosstab(testDataEC['Multistem'], pred, rownames=['Actual'], colnames=['Pred']))
            for n in compArray.index:
                #print (compArray.iloc[n][0]==1 and compArray.iloc[n][1]==1)
                result.append(compArray.iloc[n][0]==compArray.iloc[n][1])
            varCount.append(i+2)
            scoreSet.append(np.sum(result)/len(result))
        dataCollection['VariableCount'] = varCount
        dataCollection['Score'] = scoreSet
        return dataCollection
#        plt.plot(dataCollection['VariableCount'], dataCollection['Score'], '-o')
#        plt.axis([1, 13, 0.8, 1])
#        plt.title('Ratio of Correct Predictions for Hairpin Loop Formation')
#        plt.show()
    if filename == 'MPNEUM':
        for i in range(0, len(testCases)):
            X = trainData[testCases[i]]
            y = trainData['Multistem']
            clf = RandomForestClassifier(n_jobs=4)
            clf.fit(X, y)  
            pred = clf.predict(testDataMP[testCases[i]])
            compArray = pd.DataFrame(columns=['Predict','True'])
            result = []
            resetList = testDataMP.reset_index()
            compArray['Predict'] = pred
            compArray['True'] = resetList['Multistem']
            print (pd.crosstab(testDataMP['Multistem'], pred, rownames=['Actual'], colnames=['Pred']))
            for n in compArray.index:
                #print (compArray.iloc[n][0]==1 and compArray.iloc[n][1]==1)
                result.append(compArray.iloc[n][0]==compArray.iloc[n][1])
            varCount.append(i+2)
            scoreSet.append(np.sum(result)/len(result))
        dataCollection['VariableCount'] = varCount
        dataCollection['Score'] = scoreSet
        return dataCollection
#        plt.axis([1, 13, 0.8, 1])
#        plt.title('Ratio of Correct Predictions for Hairpin Loop Formation')
#        plt.plot(dataCollection['VariableCount'], dataCollection['Score'], '-o')
#        plt.show()


# In[985]:

def testHelixRF(filename):
    trainKeep = ['NucNum', 'Hairpin', 'Helix5','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5','Bulge3','Free','Tail']
    trainData = trainNotBool[trainKeep].join(dummy_ranks.ix[:, 'Base_A':])
    testDataEC = eCTestNB[trainKeep].join(dummy_ranks.ix[:, 'Base_A':])
    testDataMP = mPTestNB[trainKeep].join(dummy_ranks.ix[:, 'Base_A':])
    train_cols = ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Helix5','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5','Bulge3','Free','Tail']
    testCases=[['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U','Hairpin'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Hairpin','Helix3'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Hairpin','Helix3','HelixKnot'],
              ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Hairpin','Helix3','HelixKnot','Multistem'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Hairpin','Helix3','HelixKnot','Multistem','Internal5'],
              ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Hairpin','Helix3','HelixKnot','Multistem','Internal5','Internal3'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U','Hairpin','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5'],
              ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Hairpin','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5','Bulge3'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Hairpin','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5','Bulge3','Free'],
              ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Hairpin','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5','Bulge3','Free','Tail']]
    dataCollection = pd.DataFrame(columns=['VariableCount','Score'])
    varCount = [] 
    scoreSet = []
    if filename == 'ECOLI':
        for i in range(0, len(testCases)):
            X = trainData[testCases[i]]
            y = trainData['Helix5']
            clf = RandomForestClassifier(n_jobs=4)
            clf.fit(X, y)  
            pred = clf.predict(testDataEC[testCases[i]])
            compArray = pd.DataFrame(columns=['Predict','True'])
            result = []
            resetList = testDataEC.reset_index()
            compArray['Predict'] = pred
            compArray['True'] = resetList['Helix5']
            print (pd.crosstab(testDataEC['Helix5'], pred, rownames=['Actual'], colnames=['Pred']))
            for n in compArray.index:
                #print (compArray.iloc[n][0]==1 and compArray.iloc[n][1]==1)
                result.append(compArray.iloc[n][0]==compArray.iloc[n][1])
            varCount.append(i+2)
            scoreSet.append(np.sum(result)/len(result))
        dataCollection['VariableCount'] = varCount
        dataCollection['Score'] = scoreSet
        return dataCollection
#        plt.plot(dataCollection['VariableCount'], dataCollection['Score'], '-o')
#        plt.axis([1, 13, 0.8, 1])
#        plt.title('Ratio of Correct Predictions for Hairpin Loop Formation')
#        plt.show()
    if filename == 'MPNEUM':
        for i in range(0, len(testCases)):
            X = trainData[testCases[i]]
            y = trainData['Helix5']
            clf = RandomForestClassifier(n_jobs=4)
            clf.fit(X, y)  
            pred = clf.predict(testDataMP[testCases[i]])
            compArray = pd.DataFrame(columns=['Predict','True'])
            result = []
            resetList = testDataMP.reset_index()
            compArray['Predict'] = pred
            compArray['True'] = resetList['Helix5']
            print (pd.crosstab(testDataMP['Helix5'], pred, rownames=['Actual'], colnames=['Pred']))
            for n in compArray.index:
                #print (compArray.iloc[n][0]==1 and compArray.iloc[n][1]==1)
                result.append(compArray.iloc[n][0]==compArray.iloc[n][1])
            varCount.append(i+2)
            scoreSet.append(np.sum(result)/len(result))
        dataCollection['VariableCount'] = varCount
        dataCollection['Score'] = scoreSet
        return dataCollection
#        plt.axis([1, 13, 0.8, 1])
#        plt.title('Ratio of Correct Predictions for Hairpin Loop Formation')
#        plt.plot(dataCollection['VariableCount'], dataCollection['Score'], '-o')
#        plt.show()


# In[995]:

def testHairpinRF(filename):
    trainKeep = ['NucNum', 'Hairpin', 'Helix5','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5','Bulge3','Free','Tail']
    trainData = trainNotBool[trainKeep].join(dummy_ranks.ix[:, 'Base_A':])
    testDataEC = eCTestNB[trainKeep].join(dummy_ranks.ix[:, 'Base_A':])
    testDataMP = mPTestNB[trainKeep].join(dummy_ranks.ix[:, 'Base_A':])
    train_cols = ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Helix5','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5','Bulge3','Free','Tail']
    testCases=[['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Helix5'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Helix5','Helix3'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Helix5','Helix3','HelixKnot'],
              ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Helix5','Helix3','HelixKnot','Multistem'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Helix5','Helix3','HelixKnot','Multistem','Internal5'],
              ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Helix5','Helix3','HelixKnot','Multistem','Internal5','Internal3'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Helix5','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5'],
              ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Helix5','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5','Bulge3'],['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Helix5','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5','Bulge3','Free'],
              ['NucNum','Base_A','Base_C','Base_G','Base_N','Base_U', 'Helix5','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5','Bulge3','Free','Tail']]
    dataCollection = pd.DataFrame(columns=['VariableCount','Score'])
    varCount = [] 
    scoreSet = []
    if filename == 'ECOLI':
        for i in range(0, len(testCases)):
            X = trainData[testCases[i]]
            y = trainData['Hairpin']
            clf = RandomForestClassifier(n_jobs=4)
            clf.fit(X, y)  
            pred = clf.predict(testDataEC[testCases[i]])
            compArray = pd.DataFrame(columns=['Predict','True'])
            result = []
            resetList = testDataEC.reset_index()
            compArray['Predict'] = pred
            compArray['True'] = resetList['Hairpin']
            print (pd.crosstab(testDataEC['Hairpin'], pred, rownames=['Actual'], colnames=['Pred']))
            for n in compArray.index:
                #print (compArray.iloc[n][0]==1 and compArray.iloc[n][1]==1)
                result.append(compArray.iloc[n][0]==compArray.iloc[n][1])
            varCount.append(i+2)
            scoreSet.append(np.sum(result)/len(result))
        dataCollection['VariableCount'] = varCount
        dataCollection['Score'] = scoreSet
        return dataCollection
#        plt.plot(dataCollection['VariableCount'], dataCollection['Score'], '-o')
#        plt.axis([1, 13, 0.8, 1])
#        plt.title('Ratio of Correct Predictions for Hairpin Loop Formation')
#        plt.show()
    if filename == 'MPNEUM':
        for i in range(0, len(testCases)):
            X = trainData[testCases[i]]
            y = trainData['Hairpin']
            clf = RandomForestClassifier(n_jobs=4)
            clf.fit(X, y)  
            pred = clf.predict(testDataMP[testCases[i]])
            compArray = pd.DataFrame(columns=['Predict','True'])
            result = []
            resetList = testDataMP.reset_index()
            compArray['Predict'] = pred
            compArray['True'] = resetList['Hairpin']
            print (pd.crosstab(testDataMP['Hairpin'], pred, rownames=['Actual'], colnames=['Pred']))
            for n in compArray.index:
                #print (compArray.iloc[n][0]==1 and compArray.iloc[n][1]==1)
                result.append(compArray.iloc[n][0]==compArray.iloc[n][1])
            varCount.append(i+2)
            scoreSet.append(np.sum(result)/len(result))
        dataCollection['VariableCount'] = varCount
        dataCollection['Score'] = scoreSet
        return dataCollection
#        plt.axis([1, 13, 0.8, 1])
#        plt.title('Ratio of Correct Predictions for Hairpin Loop Formation')
#        plt.plot(dataCollection['VariableCount'], dataCollection['Score'], '-o')
#        plt.show()


# In[997]:

eCMSRF = testMultistemRF('ECOLI')


# In[998]:

eCHxRF = testHelixRF('ECOLI')


# In[999]:

eCHPRF = testHairpinRF('ECOLI')


# In[1000]:

mPMSRF = testMultistemRF('MPNEUM')


# In[1001]:

mPHxRF = testHelixRF('MPNEUM')


# In[996]:

mPHPRF = testHairpinRF('MPNEUM')


# In[973]:

plt.plot(eCHPRF['VariableCount'], eCHPRF['Score'], '-o', label='Hairpin')
plt.plot(eCHxRF['VariableCount'], eCHxRF['Score'], '-o', label='Helix5')
plt.plot(eCMSRF['VariableCount'], eCMSRF['Score'], '-o', label='Multistem')
plt.legend(loc='lower right')
plt.axis([1,13,.8,1.01])
plt.title('Ratio of Correct Predictions for Various RNA Folding Structures for E. coli')
plt.show()


# In[974]:

plt.plot(mPHPRF['VariableCount'], mPHPRF['Score'], '-^', label='Hairpin')
plt.plot(mPHxRF['VariableCount'], mPHxRF['Score'], '-^', label='Helix5')
plt.plot(mPMSRF['VariableCount'], mPMSRF['Score'], '-^', label='Multistem')
plt.legend(loc='lower right')
plt.axis([1,13,.6,1.01])
plt.title('Ratio of Correct Predictions for Various RNA Folding Structures for M. pneumoniae')
plt.show()


# In[178]:

plt.plot(eCHPRF['VariableCount'], eCHPRF['Score'], '-o', label='E. coli Hairpin')
plt.plot(eCHxRF['VariableCount'], eCHxRF['Score'], '-o', label='E. coli Helix5')
plt.plot(eCMSRF['VariableCount'], eCMSRF['Score'], '-o', label='E. coli Multistem')
plt.plot(mPHPRF['VariableCount'], mPHPRF['Score'], '-^', label='M. pneumoniae Hairpin')
plt.plot(mPHxRF['VariableCount'], mPHxRF['Score'], '-^', label='M. pneumoniae Helix5')
plt.plot(mPMSRF['VariableCount'], mPMSRF['Score'], '-^', label='M. pneumoniae Multistem')
plt.legend(loc='lower right')
plt.axis([1,13,.6,1.01])
plt.xlabel('Number of Variables')
plt.ylabel('Ratio Correct')
plt.title('Ratio of Correct Predictions for Various RNA Folding Structures')
plt.show()


# In[1010]:

from sklearn.neighbors import KNeighborsClassifier
resultsKNN = []
for n in range(1, 51, 2):
    clfKNN = KNeighborsClassifier(n_neighbors=n)
    clfKNN.fit(testDataEC['Helix5'], testDataEC['Hairpin'])
    predKNN = clfKNN.predict(testDataEC['Helix3'])
    accuracy = np.where(predKNN==test['high_quality'], 1, 0).sum() / float(len(test))
    #print ()"Neighbors: %d, Accuracy: %3f" % (n, accuracy))

    resultsKNN.append([n, accuracy])

resultsKNN = pd.DataFrame(results, columns=["n", "accuracy"])

pl.plot(results.n, results.accuracy)
pl.title("Accuracy with Increasing K")
pl.show()

#clfRF = RandomForestClassifier(n_jobs=4)
#clfRF.fit(trainData[train_cols], trainData['Hairpin'])
#preds = clfRF.predict(testDataEC[train_cols])
#pd.crosstab(testDataEC['Hairpin'], preds, rownames=['actual'], colnames=['preds'])


# In[553]:

#Create long list of all nucleotides
##Remants of first code, resulted in overwritten data...
#bases = []
#bases.extend(bBCol)
#bases.extend(bHCol)
#bases.extend(pMCol)
#bases.extend(rPCol)
#bases.extend(tMCol)


# In[380]:

#Create arrays for desired characteristics for E Coli
#Second pass of code addresses overwritten data
#dataArray = []
#b5Array = []
#b3Array = []
#hpArray = []
#hkArray = []
#h5Array = []
#h3Array = []
#msArray = []
#i5Array = []
#i3Array = []
#fArray = []
#tArray = []
#eCStrandArray = pd.DataFrame(columns=['NucNum','Base','Hairpin','Helix5','Helix3','HelixKnot','Multistem','Internal5','Internal3','Bulge5','Bulge3','Free','Tail'])


# In[381]:

#Start of test data
#E COLI and M PNEUMONIAE held out as testing sets
##Remnants of first code run, resulted in overwritten values...
#indexLen = len(bBurgStrand) + len(bHaloStrand) + len(pMultStrand) + len(rProwStrand) + len(tMariStrand)


# In[382]:

#Open files of modified ALPSX files
#Second pass of code addresses rewritten values
#with open("E Coli Stripped.txt") as textFile:
#    eclines = [line.split() for line in textFile]
#with open("M Pneumoniae Stripped.txt") as textFile:
#    mplines = [line.split() for line in textFile]
#with open("T Maritima Stripped.txt") as textFile:
#    tmlines = [line.split() for line in textFile]
#with open("R Prowazekii Stripped.txt") as textFile:
#    rplines = [line.split() for line in textFile]
#with open("P Multocida Stripped.txt") as textFile:
#    pmlines = [line.split() for line in textFile]
#with open("B Halodurans Stripped.txt") as textFile:
#    bhlines = [line.split() for line in textFile]
#with open("B Burgdorferi Stripped.txt") as textFile:
#    bblines = [line.split() for line in textFile]


# In[383]:

#Initialize lists to FALSE
#Second pass of code addresses rewritten values
#for z in range (1, len(eCCol) + 1):
#    dataArray.append(z)
#    tArray.append('FALSE')
#    fArray.append('FALSE')
#    b5Array.append('FALSE')
#    b3Array.append('FALSE')
#    i5Array.append('FALSE')
#    i3Array.append('FALSE')
#    msArray.append('FALSE')
#    h3Array.append('FALSE')
#    h5Array.append('FALSE')
#    hkArray.append('FALSE')
#    hpArray.append('FALSE')


# In[384]:

#Set end of for loop for length of grouped terms in ALPSX files
##Remnants of first code, results in overwritten data...
#groupLen = len(tmlines) + len(rplines) + len(pmlines) + len(bhlines) + len(bblines)


# In[385]:

#Create giant array of parsed text
#Must keep order of concatenated nucleotide list of: 
#indexLen = len(bBurgStrand) + len(bHaloStrand) + len(pMultStrand) + len(rProwStrand) + len(tMariStrand)
##Remnants of first code, results in overwritten data...
#giantArray = []
#giantArray.extend(bblines)
#giantArray.extend(bhlines)
#giantArray.extend(pmlines)
#giantArray.extend(rplines)
#giantArray.extend(tmlines)


# In[386]:

#Check length of giant array with groupLen
##Remnants of first code, results in overwritten data...
#len(giantArray) == groupLen


# In[387]:

#Creating categorical true and falses in data
#Second pass of code fixes overwritten values
#for x in range (0, len(eclines)):
#    lower = eclines[x][0]
#    lower = int(lower) - 1
#    upper = eclines[x][1]
#    upper = int(upper)
#    formation = eclines[x][2]
#    if formation == 'TAIL':
#        for i in range (lower, upper):
#            tArray[i] = 'TRUE'
#    elif formation == 'FREE':
#        for i in range (lower, upper):
#            fArray[i] = 'TRUE'
#    elif formation == 'BULGE5':
#        for i in range (lower, upper):
#            b5Array[i] = 'TRUE'
#    elif formation == 'BULGE3':
#        for i in range (lower, upper):
#            b3Array[i] = 'TRUE'
#    elif formation == 'INTERNAL5':
#        for i in range (lower, upper):
#            i5Array[i] = 'TRUE'
#    elif formation == 'INTERNAL3':
#        for i in range (lower, upper):
#            i3Array[i] = 'TRUE'
#    elif formation == 'MULTISTEM':
#        for i in range (lower, upper):
#            msArray[i] = 'TRUE'
#    elif formation == 'HELIX3':
#        for i in range (lower, upper):
#            h3Array[i] = 'TRUE'
#    elif formation == 'HELIX5':
#        for i in range (lower, upper):
#            h5Array[i] = 'TRUE'
#    elif formation == 'HELIXKNOT':
#        for i in range (lower, upper):
#            hkArray[i] = 'TRUE'
#    elif formation == 'HAIRPIN':
#        for i in range (lower, upper):
#            hpArray[i] = 'TRUE'


# In[388]:

#Fill in table with individual arrays
##Second pass of code address overwritten values
#eCStrandArray['NucNum'] = dataArray
#eCStrandArray['Base'] = eCCol
#eCStrandArray['Tail'] = tArray
#eCStrandArray['Free'] = fArray
#eCStrandArray['Bulge5'] = b5Array
#eCStrandArray['Bulge3'] = b3Array
#eCStrandArray['Internal5'] = i5Array
#eCStrandArray['Internal3'] = i3Array
#eCStrandArray['Multistem'] = msArray
#eCStrandArray['Helix3'] = h3Array
#eCStrandArray['Helix5'] = h5Array
#eCStrandArray['HelixKnot'] = hkArray
#eCStrandArray['Hairpin'] = hpArray


# In[ ]:



