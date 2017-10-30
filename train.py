# -*- coding: utf-8 -*-
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
import data_helper
import logging
import time,datetime

logger = logging.getLogger("test")
logger.setLevel(logging.DEBUG)
fomatter = logging.Formatter('%(asctime)s -%(filename)s-line:%(lineno)d-%(levelname)s-%(module)s:%(message)s')
fh = logging.FileHandler("log2.log")
fh.setLevel(logging.INFO)
fh.setFormatter(fomatter)
logger.addHandler(fh)


def train_task():
	#load train data
	train_X_data,train_y_data=data_helper.data_loader()
	
	#split the train data and test data
	X_train, X_test, y_train, y_test = train_test_split(train_X_data, train_y_data, \
	test_size=0.25, random_state=33)

	# new model and train model
	GBR=GradientBoostingRegressor()
	GBR=GBR.fit(X_train, y_train)

	#save model
	joblib.dump(GBR, "train_model.m")

if __name__ == '__main__':
	flag=0
	target_time = datetime.datetime(2017, 10, 29, 12, 22, 0)

	while True:
		now=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		tar_time_str=target_time.strftime('%Y-%m-%d %H:%M:%S')
		if now==tar_time_str:
			train_task()
			logger.info('train done')
			flag=1
		else:
			if flag==1:
				target_time=target_time+datetime.timedelta(minutes=1)
				logger.info('reset train time')
				flag=0
	
