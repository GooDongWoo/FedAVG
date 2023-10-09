from tensorflow.python.client import device_lib
from tensorflow.config import list_physical_devices
print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(list_physical_devices('GPU')))
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.backend import clear_session
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import random as rd
print(f"@@@@@@@@ Start time: {datetime.now()}@@@@@@@@@@@")
K_client_num=100
S_round=30  #총 라운드 수
IS_DATA_CSV_EXIST=1

#데이터(MNIST) 불러오고 전처리
#데이터 가져오고 합쳐서 70,000개로 합치고 각각 리스트로 나누고 6000개씩 뽑기
(x_train, y_train), (x_test, y_test) = mnist.load_data()
all_x=np.concatenate([x_train, x_test], axis=0)
all_y=np.concatenate([y_train, y_test], axis=0)

idx = np.argsort(all_y) #idx는 y_list에서 0부터 9까지의 라벨링된 데이터의 순서대로 정렬한 것의 np.idx_list
x_all_sorted = all_x[idx]
y_all_sorted = all_y[idx]

sorted_x_train=[]#sorted_x_train은 0~9idx에 각각 라벨링이 그것인 x_np 데이터가 들어있음.
for i in range(10):
    sorted_x_train.append(x_all_sorted[y_all_sorted == i])

'''# 각 라벨별로 몇개 있는지
print("0 :",len(sorted_x_train[0]),"1 :",len(sorted_x_train[1]),"2 :",len(sorted_x_train[2]),"3 :",len(sorted_x_train[3]),"4 :",len(sorted_x_train[4]),"5 :",len(sorted_x_train[5]),
"6 :",len(sorted_x_train[6]),"7 :",len(sorted_x_train[7]),"8 :",len(sorted_x_train[8]),"9 :",len(sorted_x_train[9]))
# 라벨0의 shape
print(x_all_sorted[y_all_sorted == 0].shape)
'''

#이제 전부 앞에 6천개만 뽑아서 x_train로 1줄로 세워버리고 전부 나머지는 테스트에다가 박아버리기.
x_train=sorted_x_train[0][:6000]
x_test=sorted_x_train[0][6000:]
for i in range(1,10):
    x_train=np.concatenate([x_train,sorted_x_train[i][0:6000:1]], axis=0)
    x_test=np.concatenate([x_test,sorted_x_train[i][6000:]], axis=0)
    
tmp_int=0
y_train=np.zeros(60000,)
for i in range(10):
    y_train[tmp_int:tmp_int+6000]=i
    tmp_int+=6000
    
tmp_int=0
y_test=np.zeros(10000,)
for i in range(10):
    y_test[tmp_int:tmp_int+(len(sorted_x_train[i])-6000)]=i
    tmp_int+=(len(sorted_x_train[i])-6000)

# 차원 변환 후, 테스트셋과 학습셋으로 나눔
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 할당하는 과정
if not(IS_DATA_CSV_EXIST):
    x_per_label=20
    data_label_list=[0,1,2,3,4,5,6,7,8,9]
    list_added_label_num=[0]*10
    last_label_left=-1# 마지막으로 1개밖에 안남은 라벨 숫자

    list_combinational=[]
    list_added_label_idx=[]
    for iter in range(x_per_label*5):
        while 1:
            if len(data_label_list)<=1:
                last_label_left=list_added_label_num[data_label_list[0]]
                what_last_num=data_label_list[0]
                break

            temp_pick2=rd.sample(data_label_list,2)

            if (list_added_label_num[temp_pick2[0]]==x_per_label) or (list_added_label_num[temp_pick2[1]]==x_per_label):
                if (list_added_label_num[temp_pick2[0]]==x_per_label):
                    data_label_list.remove(temp_pick2[0])
                if (list_added_label_num[temp_pick2[1]]==x_per_label):
                    data_label_list.remove(temp_pick2[1])
                continue
            
            list_added_label_idx.append([temp_pick2[0]*6000+list_added_label_num[temp_pick2[0]]*300,temp_pick2[1]*6000+list_added_label_num[temp_pick2[1]]*300])
            list_combinational.append(temp_pick2)

            list_added_label_num[temp_pick2[0]]+=1
            list_added_label_num[temp_pick2[1]]+=1
            break 

    #일단 나머지를 라벨 한개짜리로 채우는거
    last_label_left_set=-1
    if (last_label_left!=-1):
        last_label_left_set=(20-last_label_left)/2
        for i______ in range(int(last_label_left_set)):
            list_added_label_idx.append([what_last_num*6000+list_added_label_num[what_last_num]*300,what_last_num*6000+(list_added_label_num[what_last_num]+1)*300])
            list_added_label_num[what_last_num]+=2

    print(list_combinational)
    print(list_added_label_num)
    print(list_added_label_idx)
    print(last_label_left_set)
else:
    list_added_label_idx=[[48000, 36000], [6000, 30000], [0, 48300], [300, 30300], [24000, 30600], [48600, 24300], [12000, 30900], [42000, 12300], [42300, 24600], [31200, 24900], [18000, 54000], [600, 36300], [900, 18300], [25200, 1200], [54300, 1500], [18600, 48900], [6300, 25500], [12600, 31500], [1800, 25800], [31800, 2100], [54600, 12900], [49200, 18900], [32100, 19200], [36600, 6600], [19500, 42600], [54900, 6900], [19800, 2400], [49500, 32400], [42900, 7200], [43200, 13200], [32700, 13500], [33000, 2700], [7500, 55200], [49800, 13800], [7800, 36900], [8100, 33300], [50100, 55500], [8400, 20100], [55800, 50400], [20400, 26100], [8700, 43500], [43800, 56100], [50700, 56400], [56700, 3000], [37200, 20700], [3300, 37500], [3600, 14100], [57000, 14400], [26400, 14700], [21000, 9000], [44100, 15000], [21300, 57300], [3900, 51000], [9300, 51300], [21600, 4200], [21900, 26700], [9600, 51600], [15300, 51900], [9900, 4500], [52200, 33600], [52500, 57600], [10200, 4800], [5100, 33900], [27000, 52800], [57900, 5400], [44400, 15600], [53100, 58200], [22200, 15900], [22500, 44700], [5700, 22800], [58500, 37800], [45000, 27300], [34200, 10500], [53400, 23100], [10800, 45300], [45600, 58800], [27600, 11100], [23400, 27900], [28200, 59100], [38100, 28500], [45900, 38400], [46200, 16200], [11400, 53700], [28800, 11700], [59400, 23700], [46500, 38700], [59700, 46800], [47100, 39000], [47400, 39300], [39600, 47700], [16500, 34500], [29100, 39900], [16800, 34800], [35100, 17100], [17400, 40200], [29400, 40500], [29700, 40800], [17700, 35400], [35700, 41100], [41400, 41700]]
    print(f"#########@@@@@@@@@@@@ DATA EXIST@@@@@@@@@@@@@@#########")
    print(list_added_label_idx)    
    
print(f"@@@@@@@@ first initialized time: {datetime.now()}@@@@@@@@@@@")
before_time=datetime.now()
before_time_round=datetime.now()
after_time=datetime.now()

#env loop start>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
env_num=0
env_start_num=8
env_setting=[[10,1],[10,5],[10,20],[50,1],[50,5],[50,20],[600,1],[600,5],[600,20]]
for env_num in range(env_start_num,9):
    B_batch=env_setting[env_num][0] # 배치 사이즈
    E_epoch=env_setting[env_num][0]  # 각 클라이언트마다 몇 에포크 돌릴지
    print(f"###########B_batch={env_setting[env_num][0]}\n###########E_epoch={env_setting[env_num][0]}")
    ##서버 모델 이니셜라이징
    # 모델 구조를 설정
    server_model = Sequential()
    server_model.add(Conv2D(32, kernel_size=(5, 5), input_shape=(28, 28, 1), activation='relu'))
    server_model.add(Conv2D(64, (5, 5), activation='relu'))
    server_model.add(MaxPooling2D(pool_size=(2,2)))
    #server_model.add(Dropout(0.25))
    server_model.add(Flatten())
    server_model.add(Dense(10, activation='softmax'))

    #서버 레이어들 정보 요약
    #server_model.summary()                                                

    # 모델 실행 환경을 설정
    server_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 모델 최적화를 위한 설정 구간
    '''serverpath="./MNIST_MLP_0.hdf5"
    checkpointer = ModelCheckpoint(filepath=serverpath, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)'''
    # 모델 실행
    server_history = server_model.fit(x_train[0:2], y_train[0:2], validation_split=0.25, epochs=1, batch_size=2, verbose=0) # 최대한 학습 안할려고 2개만 학습시킴...
    #클라이언트 100명 각각 설정하는 것
    clients_model=[]
    clients_model_w=[]
    clients_path=[]
    server_w=server_model.get_weights()

    history_temp=[[],[]]
    clients_history=[]
    ##클라이언트i 모델 이니셜라이징
    clients_model=Sequential()
    clients_model.add(Conv2D(32, kernel_size=(5, 5), input_shape=(28, 28, 1), activation='relu'))
    clients_model.add(Conv2D(64, (5, 5), activation='relu'))
    clients_model.add(MaxPooling2D(pool_size=(2,2)))
    #clients_model.add(Dropout(0.25))
    clients_model.add(Flatten())
    clients_model.add(Dense(10, activation='softmax'))
    #clients_model.summary()                    
    
    # 모델 실행 환경을 설정
    clients_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 모델 최적화를 위한 설정 구간
    '''clients_path.append("./MNIST_MLP_"+str(i)+".hdf5")
    checkpointer = ModelCheckpoint(filepath=clients_path[i], monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=4)
    clients_path'''
    
    #처음 서버>>>>>>>>>>>>>>>>>클라이언트
    for i in range(0,S_round): #i equal -> per round
        #각 클라이언트들마다 computations
        clients_model_w=[]
        for j in range(K_client_num):

            clients_model.set_weights(server_w)
            #학습
            # j equal per clients
            clients_history.append(clients_model.fit(np.concatenate([x_train[list_added_label_idx[j][0]:list_added_label_idx[j][0]+300], x_train[list_added_label_idx[j][1]:list_added_label_idx[j][1]+300]], axis=0),np.concatenate([y_train[list_added_label_idx[j][0]:list_added_label_idx[j][0]+300], y_train[list_added_label_idx[j][1]:list_added_label_idx[j][1]+300]], axis=0), validation_split=0.25, epochs=E_epoch, batch_size=B_batch, verbose=0) )
            
            #clients_model_w에 각 모델들 가중치 저장
            clients_model_w.append(clients_model.get_weights())
            
            if j%10==0:
                
                print(f"{j}th client is studying diff time: {datetime.now()-before_time}-------- curr_time :{datetime.now()}")
                before_time=datetime.now()
            clear_session()
        
        #각 클라이언트들의 w를 산술평균해서 서버에다가 주는 과정 1round마다 서버<<<<<<<<<<<<<<<<<<클라이언트
        array_temp = []
        for j in range(len(clients_model_w[0])):
            array_temp.append(clients_model_w[0][j]/K_client_num)
            for k in range(1,K_client_num):
                array_temp[j]+=(clients_model_w[k][j])/K_client_num
                
        server_w=(array_temp)
        server_model.set_weights(server_w)

        #서버의 1round마다의 데이터들의 히스토리를 모으는 과정
        history_temp[1].append(server_model.evaluate(x_test, y_test)[1])
        print("@@@@@"+str(i+1)+"th Round Test Accuracy: %.4f" % (history_temp[-1][-1]))
        history_temp[0].append(server_model.evaluate(x_train, y_train)[1])
        print("@@@@@"+str(i+1)+"th Round Train Accuracy: %.4f" % (history_temp[0][-1]))
        print(f"@@@@@@@@{i}th round diff time: {datetime.now()-before_time_round}@@@@@@@@@@@@@@@@@")
        before_time_round=datetime.now()
        
    # 테스트 정확도
    print("\n Test Accuracy: %.4f" % (server_model.evaluate(x_test, y_test)[1]))
    # 검증셋과 학습셋의 오차를 저장
    y_vloss = history_temp[1] #server_history.history['val_loss']
    y_loss = history_temp[0]  #server_history.history['loss']

    # 그래프로 표현
    #plt.yticks(np.arange(0.9,1,0.01))
    x_len = np.arange(len(y_loss))
    plt.plot(x_len, y_vloss, marker='.', label=f'B={env_setting[env_num][0]} E={env_setting[env_num][1]} \nTestset_accuracy')
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:.4f}'.format(x) for x in current_values])
    #plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_accuracy')
    plt.legend(loc='lower right')
    plt.grid()
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    for x,y in zip(x_len,y_vloss):
        if(x%2==0):
            label = "{:.4f}".format(y)
            plt.annotate(label, 
                        (x,y), # x and y is the points location where we have to label
                        textcoords="offset points",
                        xytext=(0,10+11*(x%4)), # this for the distance between the points
                        ha='center',
                        arrowprops=dict(arrowstyle="->", color='green'))
    plt.savefig(f'Round={S_round} B={env_setting[env_num][0]} E={env_setting[env_num][1]}Testset_accuracy.png')
    clear_session()
    plt.show()
    np.savetxt(f'Round={S_round} B={env_setting[env_num][0]} E={env_setting[env_num][1]}Testset_accuracy.csv', np.array(y_vloss), delimiter=",", fmt="%.5f")
    break
