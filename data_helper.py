import mysql.connector
import pandas as pd
import numpy as np
import re
import datetime
import time
import time,datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib


def str_to_list(str_=''):
		rr = re.compile(r'[\d]+,[\d]+')
		match_list=rr.findall(str_)
		result=list(map(lambda x : x.split(','),match_list))
		return result

def data_sql_helper():

	# conn = mysql.connector.connect(user='root',password='123456',host='127.0.0.1',port='3306',\
	# 	database='zdb_udc')

	conn = mysql.connector.connect(user='xuyanmei1',password='xuyanmei1@',host='172.25.28.7',port='3306',\
		database='zdb_udc')
	
	cursor = conn.cursor()
	count = cursor.execute('select * from zt_traffic_prediction')

	results = cursor.fetchall()
	df_dataset=pd.DataFrame(columns=['storeId','day','preds1','preds2','starttime']) #总数据表

	preds_data=[]
	storeid_data=[]
	days_data=[]
	starttime_data=[]

	for record in results:
		index,storeid,day,preds=record

		preds=preds.decode('utf-8')
		preds_list=(str_to_list(preds))

		row_num=len(preds_list)

		preds_data.extend(preds_list)
		storeid_data.extend([storeid]*row_num)
		day=day.strftime('%Y-%m-%d')
		days_data.extend([day]*row_num)
		starttime_data.extend(range(0,24))
		
	preds_data=np.array(preds_data)

	df_dataset['storeId']=storeid_data
	df_dataset['day']=days_data

	df_dataset['preds1']=preds_data[:,0]
	df_dataset['preds2']=preds_data[:,1]
	df_dataset['starttime']=starttime_data
	conn.close()

	return df_dataset


def data_loader(train=True,args=None):

	df=data_sql_helper()

	# df=pd.read_csv('1.txt',sep=',')
	days=df['day'].drop_duplicates() # all day date in the dataset
	storeId=df['storeId'].drop_duplicates() # all day date in the dataset
	predict_values=['traffic','customer']

	target_times=range(9,23)# predict target hours value 

	train_x=[]
	train_y=[]

	

	#加载预测数据
	if not train and args != None:
		store=args['storeId']	#店铺号数据
		starttime=args['starttime']
		it=args['preds'] #需要预测的是客流量还是人流量
		day=args['day']
		t=args['starttime']#时间点
		last_day=max(days)
		# if day in days:	
			
		# sub_df=df[np.logical_and(df['day']==day,df['storeId']==store)]

		time_f=time_feature(df,day,t,store,it)
		three_hours_f=three_hours_feature(df,days,day,t,store,it,False) # three hours before feature
		week_f=week_feature(day)
		store_f=store_feature(storeId,store)
		traffic_customer_f=traffic_customer_feature(it)
		result_f=np.concatenate((time_f, three_hours_f,week_f,store_f,traffic_customer_f)).astype(float)

		return result_f

	#加载训练数据
	else:	
		for store in storeId:
			for it in predict_values:
				for da in days:
					# da: one day ,2017-08-05,...
					sub_df=df[np.logical_and(df['day']==da,df['storeId']==store)]

					all_time=sub_df['starttime'] # this day and store all hour value

					time_available=list(map(lambda x:judge(x,all_time) ,target_times))
					time_available=[x for x in time_available if x !=None]	

					for t in time_available:	
						# t is one hours value 
						time_f=time_feature(df,da,t,store,it)	# hours feature
						three_hours_f,pred_target=three_hours_feature(df,days,da,t,store,it) # three hours before feature
						week_f=week_feature(da)	# week feature
						store_f=store_feature(storeId,store)
						traffic_customer_f=traffic_customer_feature(it)
						# all feature dims 16:x1,x2....x16
						result_f=np.concatenate((time_f, three_hours_f,week_f,store_f,traffic_customer_f)).astype(float)
						# pred_target  :y
						train_x.append(result_f)
						train_y.append(pred_target)

		assert len(train_x)==len(train_y)
		return train_x,train_y


def judge(tar_time,all_time):
	tar_list=[tar_time,tar_time-1,tar_time-2,tar_time-3]
	if len(set(tar_list).difference(set(all_time)))==0:
		return tar_time

def time_feature(df,da,t,store,it):
	sub_df_befor=df[np.logical_and(df['day']<=da,df['starttime']==t)]
	sub_df_befor=sub_df_befor[df['storeId']==store]
	if it=='traffic':
		bumber_list=np.array(sub_df_befor['preds1'].tolist()).astype(float)
	elif it=='customer':
		bumber_list=np.array(sub_df_befor['preds2'].tolist()).astype(float)
	mean_=np.mean(bumber_list)
	var_=np.var(bumber_list)
	mediam_=np.median(bumber_list)
	percentile_25=np.percentile(bumber_list,25)
	percentile_75=np.percentile(bumber_list,75)

	return [mean_,var_,mediam_,percentile_25,percentile_75]


def three_hours_feature(df,days,da,tar_time,store,it,train=True):
	
	#训练调用
	if train:
		sub_df=df[np.logical_and(df['day']==da,df['storeId']==store)]
		tar_list=[tar_time,tar_time-1,tar_time-2,tar_time-3]
		num_list=[]
		for h in tar_list:
			sub_data=sub_df[np.logical_and(sub_df['starttime']==h,sub_df['storeId']==store)]
			assert sub_data.shape[0]==1 #一条数据
			if it=='traffic':
				num_list.append(sub_data['preds1'].tolist()[0])
			elif it=='customer':
				num_list.append(sub_data['preds2'].tolist()[0])


		num_list=np.array(num_list).astype(float)
		result=[np.mean(num_list[1:]),float((num_list[2]-num_list[1]))/(num_list[1]+1),\
				float((num_list[3]-num_list[2]))/(num_list[2]+1)]
		return result,num_list[0]


	else:
		num_list=[]
		sub_df=df[np.logical_and(df['day']==da,df['storeId']==store)]
		before_three_h=[tar_time-1,tar_time-2,tar_time-3]
		for h in before_three_h:
			sub_data=sub_df[sub_df['starttime']==h]
			if sub_data.shape[0]==0:
				sub_store_h=df[np.logical_and(df['starttime']==h,df['storeId']==store)]
				if it=='traffic':
					num_list.append(np.mean(np.array(sub_store_h['preds1']).astype(float)))
				elif it=='customer':
					num_list.append(np.mean(np.array(sub_store_h['preds2']).astype(float)))
			else:
				if it=='traffic':
					num_list.append(sub_data['preds1'].tolist()[0])
				elif it=='customer':
					num_list.append(sub_data['preds2'].tolist()[0])

		num_list=np.array(num_list).astype(float)
		result=[np.mean(num_list),float((num_list[1]-num_list[0]))/(num_list[0]+1),\
				float((num_list[2]-num_list[1]))/(num_list[1]+1)]
		return result




def week_feature(da):
	da_int_list=[int(x) for x in da.split('-')]
	day = datetime.datetime(*da_int_list).weekday()
	one_hot=np.zeros(7)
	one_hot[day]=1.0
	return one_hot

def store_feature(storeId,store):
	zero_enc=np.zeros(len(storeId)).tolist()
	zero_enc[storeId.tolist().index(store)]=1
	return zero_enc


def traffic_customer_feature(it):
	if it=='traffic' :
		return [1,0]
	elif it=='customer':
		return [0,1]