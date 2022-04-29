from typing import NamedTuple
import numpy
def load_data(log_folder:str)->NamedTuple('Outputs', [('start_time_string',str)]):
    import numpy as np
    import time
    import sys
    print("import done...")
    start = time.time()
    data= np.load("triplet-data.npz")
    sys.path.append("./")
    
#    from config import img_size, channel, faces_data_dir, FREEZE_LAYERS, classify, facenet_weight_path   
#    from inception_resnet_v1 import InceptionResNetV1
#    from utils import scatter 
    X_train, X_test = data['arr_0'], data['arr_1']
    print(X_train.shape, X_test.shape)
    print("Saving data...")
    #print(X_train)
    #print(X_test)
    np.savez_compressed('/persist-log/triplet-data.npz', X_train, X_test)
    print('Save complete ...')
    start_time_string=str(start) #type is string
    return [start_time_string]


def distributed_training_worker1(start_time_string:str)->NamedTuple('Outputs',[('model_path',str)]):
    import numpy as np
    import sys
    import time
    import tensorflow as tf
    import json
    import os
    sys.path.append("./")
    sys.path.append("/persist-log")
    from config import img_size, channel, faces_data_dir, FREEZE_LAYERS, classify, facenet_weight_path
    from inception_resnet_v1 import InceptionResNetV1
    from itertools import permutations
    from tqdm import tqdm
    from tensorflow.keras import backend as K
    from sklearn.manifold import TSNE
    
    #load data from pvc in the container
    data = np.load('/persist-log/triplet-data.npz')
    X_train, X_test = data['arr_0'], data['arr_1']

    def training_model(in_shape,freeze_layers,weights_path):

        def create_base_network(in_dims,freeze_layers,weights_path):
            model = InceptionResNetV1(input_shape=in_dims, weights_path=weights_path)
            print('layer length: ', len(model.layers))
            for layer in model.layers[:freeze_layers]:
                layer.trainable = False
            for layer in model.layers[freeze_layers:]:
                layer.trainable = True
            return model
        
        def triplet_loss(y_true,y_pred,alpha=0.4):
            total_lenght = y_pred.shape.as_list()[-1]
            anchor = y_pred[:, 0:int(total_lenght * 1 / 3)]
            positive = y_pred[:, int(total_lenght * 1 / 3):int(total_lenght * 2 / 3)]
            negative = y_pred[:, int(total_lenght * 2 / 3):int(total_lenght * 3 / 3)]
            # distance between the anchor and the positive
            pos_dist = K.sum(K.square(anchor - positive), axis=1)
            # distance between the anchor and the negative
            neg_dist = K.sum(K.square(anchor - negative), axis=1)
            # compute loss
            basic_loss = pos_dist - neg_dist + alpha
            loss = K.maximum(basic_loss, 0.0)
            return loss
        # define triplet input layers
        anchor_input = tf.keras.layers.Input(in_shape, name='anchor_input')
        positive_input = tf.keras.layers.Input(in_shape, name='positive_input')
        negative_input = tf.keras.layers.Input(in_shape, name='negative_input')
        Shared_DNN = create_base_network(in_shape, freeze_layers, weights_path)
        # Shared_DNN.summary()
        # encoded inputs
        encoded_anchor = Shared_DNN(anchor_input)
        encoded_positive = Shared_DNN(positive_input)
        encoded_negative = Shared_DNN(negative_input)
        # output
        merged_vector = tf.keras.layers.concatenate([encoded_anchor, encoded_positive, encoded_negative],axis=-1,name='merged_layer')
        model = tf.keras.Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_vector)
        model.compile(
            optimizer=adam_optim,
            loss=triplet_loss,
        )
        return model
    
    
    os.environ['TF_CONFIG'] = json.dumps({'cluster': {'worker': ["pipeline-worker-1:3000","pipeline-worker-2:3000","pipeline-worker-3:3000"]},'task': {'type': 'worker', 'index': 0}})
    #os.environ['TF_CONFIG'] = json.dumps({'cluster': {'worker': ["pipeline-worker-1:3000"]},'task': {'type': 'worker', 'index': 0}})


    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        tf.distribute.experimental.CollectiveCommunication.RING)
    NUM_WORKERS = strategy.num_replicas_in_sync
    print('=================\r\nWorkers: ' + str(NUM_WORKERS) + '\r\n=================\r\n')
    learn_rate = 0.0001 + NUM_WORKERS * 0.00006
    adam_optim = tf.keras.optimizers.Adam(lr=learn_rate)
    batch_size = 32* NUM_WORKERS
    model_path='/persist-log/weight_tfdl.h5'
    print(model_path)
    callbacks = [tf.keras.callbacks.ModelCheckpoint(model_path, save_weights_only=True, verbose=1)]
    #X_train=np.array(X_train)
    #print(type(X_train))
    with strategy.scope():
        Anchor = X_train[:, 0, :].reshape(-1, img_size, img_size, channel)
        Positive = X_train[:, 1, :].reshape(-1, img_size, img_size, channel)
        Negative = X_train[:, 2, :].reshape(-1, img_size, img_size, channel)
        Y_dummy = np.empty(Anchor.shape[0])
        model = training_model((img_size, img_size, channel), FREEZE_LAYERS, facenet_weight_path)
        
    model.fit(x=[Anchor, Positive, Negative],
        y=Y_dummy,
        # Anchor_test = X_test[:, 0, :].reshape(-1, img_size, img_size, channel)
        # Positive_test = X_test[:, 1, :].reshape(-1, img_size, img_size, channel)
        # Negative_test = X_test[:, 2, :].reshape(-1, img_size, img_size, channel)
        # Y_dummy = np.empty(Anchor.shape[0])
        # Y_dummy2 = np.empty((Anchor_test.shape[0], 1))
        # validation_data=([Anchor_test,Positive_test,Negative_test],Y_dummy2),
        # validation_split=0.2,
        batch_size=batch_size,  # old setting: 32
        # steps_per_epoch=(X_train.shape[0] // batch_size) + 1,
        epochs=10,
        callbacks=callbacks
        )  
    end = time.time()
    start_time_float=float(start_time_string)
    print('execution time = ', ((end - start_time_float)/60))
    return [model_path]


def model_prediction(model_path:str)->NamedTuple('Outputs',[('model_path',str)]):
    from os import listdir
    from os.path import isfile
    import time
    import numpy as np
    import cv2
    from sklearn.manifold import TSNE
    from scipy.spatial import distance
    import tensorflow as tf
    import sys
    sys.path.append("./")
    sys.path.append("/persist-log")
    sys.path.append("/facenet/test")
    from img_process import align_image, prewhiten
    from triplet_training import create_base_network
    from utils import scatter
    from config import img_size, channel, classify, FREEZE_LAYERS, facenet_weight_path, faces_data_dir
    anchor_input = tf.keras.Input((img_size, img_size, channel,), name='anchor_input')
    Shared_DNN = create_base_network((img_size, img_size, channel), FREEZE_LAYERS, facenet_weight_path)
    encoded_anchor = Shared_DNN(anchor_input)
    
    model = tf.keras.Model(inputs=anchor_input, outputs=encoded_anchor)
    model.load_weights(model_path)
    model.summary()
    start = time.time()
    def l2_normalize(x, axis=-1, epsilon=1e-10):
        output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
        return output


    # Acquire embedding from image
    def embedding_extractor(img_path):
        img = cv2.imread(img_path)
        aligned = align_image(img)
        #cv2.imwrite("facenet/align/"+"_aligned.jpg", aligned)
        if aligned is not None:
            aligned = aligned.reshape(-1, img_size, img_size, channel)

            embs = l2_normalize(np.concatenate(model.predict(aligned)))
            return embs
        else:
            print(img_path + ' is None')
            return None
        
    testset_dir = 'facenet/test/'
    items = listdir(testset_dir)

    jpgsList = [x for x in items if isfile(testset_dir + x)]
    foldersList = [x for x in items if not isfile(testset_dir + x)]

    print(jpgsList)
    print(foldersList)

    acc_total = 0
    for i, anch_jpg in enumerate(jpgsList):
        anchor_path = testset_dir + anch_jpg
        anch_emb = embedding_extractor(anchor_path)

        for j, clt_folder in enumerate(foldersList):
            clt_path = testset_dir + clt_folder + '/'
            clt_jpgs = listdir(clt_path)
            #print('anchor_path is :',anchor_path)
            #print('clt_jpgs is :',clt_jpgs)
            #print('clt_path is :',clt_path)

            str = anch_jpg
            computeType = 1 if clt_folder == str.replace('.jpg', '') else 0

            loss = 0
            if computeType == 1:
                sum1 = 0
                print('==============' + clt_folder + '&' + anch_jpg + '==============')
                for k, clt_jpg in enumerate(clt_jpgs):
                    clt_jpg_path = clt_path + clt_jpg
                    clt_emb = embedding_extractor(clt_jpg_path)
                    distanceDiff = distance.euclidean(anch_emb, clt_emb)  # calculate the distance
                    #print('distance = ', distanceDiff)
                    sum1 = distanceDiff + sum1
                    loss = loss + 1 if distanceDiff >= 1 else loss

                print("sum1", sum1 / 50.0)
                print('loss: ', loss)
                accuracy = (len(clt_jpgs) - loss) / len(clt_jpgs)
                print('accuracy: ', accuracy)
                acc_total += accuracy
            else:
                print('==============' + clt_folder + '&' + anch_jpg + '==============')
                sum2 = 0
                for k, clt_jpg in enumerate(clt_jpgs):
                    clt_jpg_path = clt_path + clt_jpg
                    clt_emb = embedding_extractor(clt_jpg_path)
                    distanceDiff = distance.euclidean(anch_emb, clt_emb)  # calculate the distance
                    #print('distance = ', distanceDiff)
                    loss = loss + 1 if distanceDiff < 1 else loss
                    sum2 = distanceDiff + sum2
                print("sum2", sum2 / 50.0)
                print('loss: ', loss)
                accuracy = (len(clt_jpgs) - loss) / len(clt_jpgs)
                print('accuracy: ', accuracy)
                acc_total += accuracy

            print('--acc_total', acc_total)

    acc_mean = acc_total / 81 * 100
    print('final acc++------: ', acc_mean)
    end = time.time()
    print ('execution time', (end - start))
    
    return [model_path]

#serving
def serving(model_path:str, log_folder:str):
    from flask import Flask,render_template,url_for,request,redirect,make_response,jsonify
    from werkzeug.utils import secure_filename
    import os 
    import cv2
    import sys
    import time
    import base64
    import math
    from datetime import timedelta
    import numpy as np
    from os import listdir
    from os.path import isfile
    from sklearn.manifold import TSNE
    from scipy.spatial import distance
    import tensorflow as tf
    
    sys.path.append("./")
    sys.path.append("/persist-log")
    sys.path.append("/templates")
    
    from img_process import align_image, prewhiten
    from triplet_training import create_base_network
    from utils import scatter
    from config import img_size, channel, classify, FREEZE_LAYERS, facenet_weight_path, faces_data_dir
    serving_time = time.time
    ALLOWED_EXTENSIONS = set(['jpg','JPG'])
    

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS


    def return_img_stream(img_local_path):
        img_stream = ''
        with open(img_local_path,'rb') as img_f:
            img_stream = img_f.read()
            img_stream = base64.b64encode(img_stream).decode()
        return img_stream

        # L2 normalization
    def l2_normalize(x, axis=-1, epsilon=1e-10):
        output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
        return output
    
#--------------------------------------------------------------demo.py 

    # Acquire embedding from image
    def embedding_extractor(img_path,model):
        img = cv2.imread(img_path)
        aligned = align_image(img)
        #cv2.imwrite("facenet/align/"+"_aligned.jpg", aligned)
        if aligned is not None:
            aligned = aligned.reshape(-1, img_size, img_size, channel)

            embs = l2_normalize(np.concatenate(model.predict(aligned)))
            return embs
        else:
            print(img_path + ' is None')
            return None
#-------------------------------------------------------------flask

    app = Flask(__name__, template_folder="/templates")

    app.send_file_max_age_default = timedelta(seconds=1)
    
    @app.route('/upload',methods=['GET','POST'])

    def upload():
        img_stream = ''
        loss = 0
        distanceDiffbig = 0
        distanceDiffsmall = 0
        distance_sum = 0

        face = ''
        face2 = ''
        face3 = ''
        acc_mean = 0

        distance_low1 = 0
        distance_low2 = 0
        distance_low3 = 0
        distance_show1 = 2
        distance_show2 = 2
        distance_show3 = 2
        
        if request.method =='POST':
            f = request.files['file']
            user_input = request.form.get('name')
            basepath = os.path.dirname(__file__)
            sys.path.append('/facenet/test')
            upload_path = os.path.join(basepath,'/facenet/test',secure_filename(f.filename))
            print(basepath)
            f.save(upload_path)
            #start = time.time()
            #model_path = '/persist-log/weight_tfdl.h5'
            anchor_input = tf.keras.Input((img_size, img_size, channel,), name='anchor_input')
            Shared_DNN = create_base_network((img_size, img_size, channel), FREEZE_LAYERS, facenet_weight_path)
            encoded_anchor = Shared_DNN(anchor_input)

            model = tf.keras.Model(inputs=anchor_input, outputs=encoded_anchor)
            model.load_weights(model_path) #/persist-log
            model.summary()

            testset_dir = 'facenet/test/'
            items = listdir(testset_dir)

            jpgsList = [x for x in items if isfile(testset_dir + x)]
            foldersList = [x for x in items if not isfile(testset_dir + x)]

            print(jpgsList)
            print(foldersList)
            
            acc_total = 0
            img_stream = return_img_stream(upload_path)
            for i, anch_jpg in enumerate(jpgsList):
                #anchor_path = testset_dir + anch_jpg
                anch_emb = embedding_extractor(upload_path,model)
                
                for j, clt_folder in enumerate(foldersList):
                    clt_path = testset_dir + clt_folder + '/'
                    clt_jpgs = listdir(clt_path)
                    str = anch_jpg
                    print('==============' + clt_folder + '&' + anch_jpg + '==============')
    
                    for k, clt_jpg in enumerate(clt_jpgs):
                        clt_jpg_path = clt_path + clt_jpg
                        clt_emb = embedding_extractor(clt_jpg_path,model)
                        distanceDiff = distance.euclidean(anch_emb, clt_emb)  # calculate the distance
                        distance_sum=distance_sum + distanceDiff
    
                        if distanceDiff >= 1:
                            distanceDiffbig = distanceDiffbig + 1
                    
                        else:
                            distanceDiffsmall = distanceDiffsmall + 1
                        
                        if distanceDiffbig >= distanceDiffsmall :
                            loss = distanceDiffsmall
                        
                        else:
                            loss = distanceDiffbig
                            
                    distance_sum=distance_sum / 16  
                
                    if distance_sum < distance_show3: 
    
                        if distance_sum < distance_show2:
            
                            if distance_sum < distance_show1:
                                distance_show1 = distance_sum
                                distance_low1 = distance_sum
                                face =  clt_folder
                            
                            else:
                                distance_low2 = distance_sum
                                distance_show2 = distance_sum
                                face2 =  clt_folder
                    
                        else:
                            distance_show3 = distance_sum
                            distance_low3 = distance_sum
                            face3 = clt_folder
                    else:
                        distanceDiff = distanceDiff
                        
                    print('distance sum is:', distance_sum)
                    print('distanceDiffsmall = ', distanceDiffsmall)
                    print('distanceDiffbig = ', distanceDiffbig)
                    print( distanceDiff)
                    
                    distance_sum = 0
                    distanceDiffsmall = 0
                    distanceDiffbig = 0
                    print('loss: ', loss)
                    accuracy = (len(clt_jpgs) - loss) / len(clt_jpgs)
                    acc_total += accuracy
                    print('face = ', face)
                    print('The first is:',face,'distance is ',distance_low1)
                    print('The Second is:',face2,'distance is ',distance_low2)
                    print('The third is:',face3,'distance is ',distance_low3)
                    
                distance_low1 = round(distance_low1,2) 
                distance_low2 = round(distance_low2,2)
                distance_low3 = round(distance_low3,2)
            acc_mean = acc_total / 9 * 100
            acc_mean = round(acc_mean,2)
            print('final acc++------: ', acc_mean)
            os.remove(upload_path)
            #end = time.time()
            #print ('execution time', (end - serving_time))
            
        return render_template('upload.html',img_stream = img_stream, face = face , face2 = face2 , face3 = face3 , distance_low1 = distance_low1, distance_low2 = distance_low2 , distance_low3 = distance_low3, acc_mean = acc_mean )
    
    if __name__ == '__main__':
        app.run(host = '127.0.0.1',port=8987,debug=True)
        
    return


import kfp.dsl as dsl
import kfp.components as components
from typing import NamedTuple
import kfp
from kfp.components import func_to_container_op, InputPath, OutputPath
from kubernetes.client.models import V1ContainerPort
@dsl.pipeline(
   name='triplet_training pipeline',
   description='triplet training test.'
)
def triplet_training_pipeline():

    log_folder = '/persist-log'
    pvc_name = "task-pv-claim"
    
    #label name
    name="pod-name"
    value1="worker-1" # selector pod-name: worker-1
    value2="worker-2" # selector pod-name: worker-2
    value3="worker-3" # selector pod-name: worker-3
    
    container_port=3000
    
    #select node
    label_name="disktype"
    label_value1="worker-1"   
    label_value2="worker-2"   
    label_value3="worker-3"   
    
    vop = dsl.VolumeOp(
        name=pvc_name,
        resource_name="newpvc",
        storage_class="standard",
        size="3Gi",
        modes=dsl.VOLUME_MODE_RWM
    )
    
    load_data_op=func_to_container_op(
        func=load_data,
        base_image="mike0355/k8s-facenet-distributed-training:4",  
    )
        
    distributed_training_worker1_op=func_to_container_op(
        func=distributed_training_worker1,
        base_image="mike0355/k8s-facenet-distributed-training:4"
    )
    
    # distributed_training_worker2_op=func_to_container_op(
    #     func=distributed_training_worker2,
    #     base_image="mike0355/k8s-facenet-distributed-training:4"
    # )
    
    # distributed_training_worker3_op=func_to_container_op(
    #     func=distributed_training_worker3,
    #     base_image="mike0355/k8s-facenet-distributed-training:4"
    # )  
    
    model_prediction_op=func_to_container_op(
        func=model_prediction,
        base_image="mike0355/k8s-facenet-distributed-training:4"
    )
    
    serving_op=func_to_container_op(
        func=serving,
        base_image="mike0355/k8s-facenet-serving:3"
    )

  #----------------------------------------------------------task  
    load_data_task=load_data_op(log_folder).add_pvolumes({
        log_folder:vop.volume,
    })
    
    distributed_training_worker1_task=distributed_training_worker1_op(load_data_task.outputs['start_time_string']).add_pvolumes({  #woker1
        log_folder:vop.volume,
    }).add_pod_label(name,value1)
        
    
    # distributed_training_worker2_task=distributed_training_worker2_op(load_data_task.outputs['start_time_string']).add_pvolumes({  #woker2
    #     log_folder:vop.volume,
    # }).add_pod_label(name,value2).add_port(V1ContainerPort(container_port=3000,host_port=3000)).add_node_selector_constraint(label_name,label_value2)
    
    
    # distributed_training_worker3_task=distributed_training_worker3_op(load_data_task.outputs['start_time_string']).add_pvolumes({  #woker3
    #     log_folder:vop.volume,
    # }).add_pod_label(name,value3).add_port(V1ContainerPort(container_port=3000,host_port=3000)).add_node_selector_constraint(label_name,label_value3)
    
         
    model_prediction_task=model_prediction_op(distributed_training_worker1_task.outputs['model_path']).add_pvolumes({
        log_folder:vop.volume,
    })
    
    
    serving_task=serving_op(model_prediction_task.outputs['model_path'], log_folder).add_pvolumes({
        log_folder:vop.volume,
    })
 
kfp.compiler.Compiler().compile(triplet_training_pipeline, 'distributed-training-1011-final.yaml')