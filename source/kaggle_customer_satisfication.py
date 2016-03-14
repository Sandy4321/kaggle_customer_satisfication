'''
Created on Mar 4, 2016

@author: richard
'''
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.metrics import roc_curve, auc
import operator
from mrmr import MRMR_LOGGER, DiscreteMrmr, FastCaim, UiThread, GaussianKde, MixedMrmr, PhyloMrmr

def create_feature_map(features, filename):
    outfile = open(filename, 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()   

def compute_jaccard_index(set_1, set_2):
    n = len(set_1.intersection(set_2))
    return n / float(len(set_1) + len(set_2) - n)

def order_features_by_gains(bst, feature_map_file):
    str_dump = bst.get_dump(feature_map_file,with_stats=True)
    
    tree_arr = []
    for i_tree, tree in enumerate(str_dump):
        arr_lvls=tree.split('\n\t')
        a_tree = {}
        for lvl in arr_lvls:
            a_lvl ={}
            dum1 = lvl.split(',')
            if('leaf' in lvl):
                dum1[0].replace('\t','')
                dum10 = dum1[0].split(':')
                lvl_id = int(dum10[0])
                dum11 = dum10[1].split('leaf=')
                leaf = float(dum11[1])
                
                cover = float(dum1[1].replace('\n','').split('cover=')[1])
                a_lvl['lvl_id']=lvl_id
                a_lvl['leaf']=leaf
                a_lvl['cover']=cover
            else:
                dum10 = dum1[0].replace('\t','').replace('\n','')
                dum11 = dum10.split(':')
                lvl_id = int(dum11[0])
                dum12 = dum11[1].split('yes=')
                dum13 = dum12[0].replace('[','').replace(']','').split('<')
                feat_name = dum13[0]
                
                yes_to = int(dum12[1])
                no_to = int(dum1[1].split('no=')[1])
                missing = int(dum1[2].split('missing=')[1])
                gain = float(dum1[3].split('gain=')[1])
                cover = float(dum1[4].split('cover=')[1])            
                feat_thr = float(dum12[1])
                
                a_lvl['lvl_id']=lvl_id
                a_lvl['feat_name']=feat_name
                a_lvl['feat_thr'] = feat_thr
                a_lvl['yes_to'] = yes_to
                a_lvl['no_to']=no_to
                a_lvl['missing'] = missing
                a_lvl['gain']=gain
                a_lvl['cover']=cover
                
            a_tree[str(lvl_id)] = a_lvl
        tree_arr.append(a_tree)    
    feat_vocabulary = {}
    for tree in tree_arr:
        for lvl in tree:
            if('gain' in tree[lvl]):
                feat_data = feat_vocabulary.setdefault(tree[lvl]['feat_name'],{'gain':tree[lvl]['gain'],'cover':tree[lvl]['cover']})
                if(cmp(feat_data,{'gain':tree[lvl]['gain'],'cover':tree[lvl]['cover']})<>0):
                    try:
                        feat_vocabulary[tree[lvl]['feat_name']]['gain'] += tree[lvl]['gain']                    
                        feat_vocabulary[tree[lvl]['feat_name']]['cover'] += tree[lvl]['cover']
                    except:
                        feat_vocabulary[tree[lvl]['feat_name']]['gain'] = tree[lvl]['gain']                    
                        feat_vocabulary[tree[lvl]['feat_name']]['cover'] = tree[lvl]['cover']          
    
    sorted_feats = sorted(feat_vocabulary.items(),key=lambda k:k[1]['gain'], reverse=True)
    return sorted_feats


file_path = '/home/richard/data_repo/Kaggle_customer_satisfication/'
all_df  = pd.read_csv(file_path + 'train.csv', header=0)
feats=list(all_df.columns.values)[1:-1]

msk = np.random.rand(len(all_df)) <= 0.7
train_df = all_df[msk]
test_df = all_df[~msk]

test_ID = test_df['ID']
test_Y = test_df['TARGET']
test_X = test_df[feats].as_matrix()

train_X = train_df[feats].as_matrix()
train_Y = train_df['TARGET']
train_ID = train_df['ID']

xg_train = xgb.DMatrix( train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)

f = open(file_path + 'xg_cus_satisf.csv','w')
f.write('FP_Rate,TP_Rate,ROC_AUC,Cut_Off,,,eta,num_feature,gamma,min_child_wgt,max_depth,max_delta,colsample\n')
f_predict = open(file_path + '/xg_predictions_cus_satisf.csv','w')
f_predict.write('MD5,Prediction,Truth\n')

eta_opt= [0.003] # 3
max_depth_opt = [30] # 4
gamma_opt = [0.1] #np.arange(0.8,1.0,0.2) # 2
subsamp_opt = [1]    
num_feature_opt = [200]
delta_opt = [2] # 4
child_weight_opt = [2]# 3
colsample_opt = [1] # 3

roc_max = 0
tpr_max = 0
fpr_max = 0
thr_max = 0
optim_param = {}
round=0
best_prediction =[]

for a_depth in max_depth_opt:
    for a_eta in eta_opt:
        for nm_f in num_feature_opt:
            for a_gamma in gamma_opt:
                for a_delta in delta_opt:
                    for a_childwgt in child_weight_opt:
                        for a_colsample in colsample_opt:
                
                            param = {}
                            # use softmax multi-class classification
                            param['booster'] = 'gbtree' # 'gbtree'
                            param['alpha'] = 2
                            param['lambda'] = 1
                            param['num_feature'] = nm_f
                            param['silent'] = 1
                            param['objective'] = 'binary:logistic' #'multi:softmax'
                            # scale weight of positive examples
                            param['eta'] = a_eta
                            param['gamma'] = a_gamma                                
                            param['max_depth'] = a_depth                                
                            param['min_child_weight']=a_childwgt
                            param['max_delta_step']=a_delta
                            param['colsample_bytree']=a_colsample
                            param['save_period']=0
                            param['subsample']=1
                            watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
                            num_round = 300
                            bst = xgb.train(param, xg_train, num_round, watchlist,verbose_eval=True );
                            # get prediction
                            predictions = bst.predict( xg_test );
                            
                            
                            # Compute ROC curve and ROC area for each class
                            
                            n_classes = 2
                            fpr, tpr, thr = roc_curve(test_Y, predictions)
                            roc_auc = auc(fpr, tpr)
                            cut_off_indx=next(x[0] for x in enumerate(fpr) if x[1] >= 0.2)
                            if(cut_off_indx>1):
                                cut_off_indx-=1
                            
                            tpr_now = tpr[cut_off_indx]
                            thr_now = thr[cut_off_indx]
                            fpr_now = fpr[cut_off_indx]
#                                     print("comb %d, roc_now is: %f, cutoff_now is: %f, fpr_now is: %f, tpr_now is: %f with eta=%f, num_feature=%d, gamma=%f,min_child_wgt=%f, max_depth=%d, max_delta=%f, colsample=%f" % (round, roc_auc, thr_now, fpr_now, tpr_now, a_eta,nm_f, a_gamma, a_childwgt, a_depth, a_delta,a_colsample),file=f)
#                                 f.write("%s,%f,%f,%f,%f,,,%f,%f,,%f,%f,%f,%f,%f,%f,%f, comb %d\n" % (f_type, fpr_max,tpr_max,roc_max,thr_max,FP_rates[mime_i], TP_rates[mime_i],a_eta,nm_f, a_gamma, a_childwgt, a_depth, a_delta,a_colsample,round ))
#                                 print("comb %d, roc_now is: %f, cutoff_now is: %f, fpr_now is: %f, tpr_now is: %f with eta=%f, num_feature=%d, gamma=%f,min_child_wgt=%f, max_depth=%d, max_delta=%f, colsample=%f" % (round, roc_auc, thr_now, fpr_now, tpr_now, a_eta,nm_f, a_gamma, a_childwgt, a_depth, a_delta,a_colsample))
                            round = round + 1
                            if(tpr_now>tpr_max):
#                                     if(roc_auc>roc_max):
                                roc_max = roc_auc   
                                tpr_max = tpr_now
                                fpr_max = fpr_now
                                thr_max = thr_now
                                best_prediction = predictions.tolist()
#                                         print("locally, roc_max is: %f, cutoff_now is: %f, fpr_now is: %f, max_tpr now is: %f, with eta=%f, num_feature=%d, gamma=%f,min_child_wgt=%f, max_depth=%d, max_delta=%f, colsample=%f" % (roc_max, thr_max, fpr_max, tpr_max, a_eta,nm_f, a_gamma, a_childwgt, a_depth, a_delta,a_colsample),file=f)
                                f.write("%f,%f,%f,%f,,,%f,%f,%f,%f,%f,%f,%f, local_max\n" % (fpr_max,tpr_max,roc_max,thr_max,a_eta,nm_f, a_gamma, a_childwgt, a_depth, a_delta,a_colsample ))
#                                 print("locally, roc_max is: %f, cutoff_now is: %f, fpr_now is: %f, max_tpr now is: %f with eta=%f, num_feature=%d, gamma=%f,min_child_wgt=%f, max_depth=%d, max_delta=%f, colsample=%f" % (roc_max, thr_max, fpr_max, tpr_max, a_eta,nm_f, a_gamma, a_childwgt, a_depth, a_delta,a_colsample))
                                for i,p in enumerate(best_prediction):                                        
                                    f_predict.write('%s,%f,%d\n' % (test_ID.iloc[i], p, test_Y.iloc[i]))    
#                                 optim_param=param.copy()
    
# a_str = ("%s,%f,%f,%f,%f,,,%f,%f,%f,%f,%f,%f,%f, final_max\n"  % ( fpr_max,tpr_max,roc_max,thr_max, optim_param['eta'],optim_param['num_feature'], optim_param['gamma'],
#                                                                                                                                optim_param['min_child_weight'], optim_param['max_depth'], optim_param['max_delta_step'],optim_param['colsample_bytree']))
l_pred = map(lambda x: 1 if x>thr_max else 0, best_prediction)
from sklearn.metrics import confusion_matrix
from sklearn import metrics
cm = confusion_matrix(test_Y, l_pred)
accur = (cm[0,0]+cm[1,1])/float(len(l_pred))
precision = metrics.precision_score(test_Y, l_pred)
recall_tpr = metrics.recall_score(test_Y, l_pred)
FPR = cm[0,1]/float(cm[0,1]+cm[0,0]) 
FNR = cm[1,0]/float(cm[1,0]+cm[1,1])
TNR = cm[0,0]/float(cm[0,0]+cm[0,1])
create_feature_map(feats,file_path)
importance = bst.get_fscore(fmap=file_path+'xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1),reverse=True)
bst.dump_model(file_path + 'xgb.model.dump', with_stats = True)

# f.write(a_str)
f.write('\n TN: %d, FP: %d, FN: %d, TP: %d ' %(cm[0,0], cm[0,1], cm[1,0], cm[1,1]))
f.write('\n Accuracy: %f, Precision: %f, TPR: %f, FPR: %f, FNR: %f, TNR: %f' %(accur, precision, recall_tpr, FPR, FNR, TNR))
for feat in importance[:50]:
    f.write('\n %s, %f' % (feat[0],feat[1]))
# print(a_str)    
print('\n Accuracy: %f, Precision: %f, TPR: %f, FPR: %f, FNR: %f, TNR: %f' %(accur, precision, recall_tpr, FPR, FNR, TNR))
feature_map_file = file_path+'xgb.fmap'
sorted_feats = order_features_by_gains(bst,feature_map_file)
[x[0] for x in sorted_feats]

# THRESHOLD = 0.7
# normalized = False
# 
# labels = train_df['TARGET'].values
# data = train_df[feats].as_matrix()  # only use training data for feature selection
# klass = DiscreteMrmr
# num_features = 50
# targets = labels.astype(bool)
# variables = data.astype(float)
# nrow, ncol = variables.shape
# selector = klass(num_features, klass.MID, THRESHOLD)
#    
# # b = time.time()
# ui = None
# maxrel, mrmr = selector._mrmr_selection(num_features, klass.MID, variables, targets, threshold=THRESHOLD, ui=ui)

cut_X = pd.qcut(train_X, 20,labels=False, retbins=True)
from skfeature.function.similarity_based import fisher_score
score = fisher_score.fisher_score(train_X, train_Y)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

bestFeat = SelectKBest()
bestFeat.fit(train_X, train_Y)
feat_scr = zip(feats,bestFeat.scores_)
feat_scr = [f for f in feat_scr if not np.isnan(f[1])]
sorted_fetas = sorted(feat_scr, key=lambda k:k[1], reverse=True)

# estimator = SVR(kernel="linear")
# selector = RFE(estimator, 5, step=1)
# selector.fit(train_X, train_Y)  # slow

from sklearn.ensemble import GradientBoostingClassifier
g_cls = GradientBoostingClassifier(n_estimators=10)
g_cls.fit(train_X, train_Y)
g_feats = g_cls.feature_importances_
g_feat_scr = zip(feats,g_feats)
g_feat_scr = [f for f in g_feat_scr if not np.isnan(f[1])]
g_sorted_fetas = sorted(g_feat_scr, key=lambda k:k[1], reverse=True)


 
from skfeature.function.information_theoretical_based import FCBF, LCSI, MRMR, JMI
score = FCBF.fcbf(train_X, train_Y) 
fcbf_sorted= [feats[i] for i in score]

score = MRMR.mrmr(train_X, train_Y, n_selected_features = 50) 
MRMR_sorted= [feats[i] for i in score]

score = JMI.jmi(train_X, train_Y, n_selected_features = 50) 
JMI_sorted= [feats[i] for i in score]


