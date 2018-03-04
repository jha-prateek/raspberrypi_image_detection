import tensorflow as tf
import cv2

cam = cv2.VideoCapture(0)

label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("./tf_files/retrained_labels.txt")]

with tf.gfile.FastGFile("./tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    # while True:
    ret, frame = cam.read()
    frame = cv2.resize(frame, (480,640))
    # cv2.imshow("frame", frame)
    predictions = sess.run(softmax_tensor, {'DecodeJpeg:0': frame})
    predictions = predictions[0].tolist()
    max_value = max(predictions)
    max_index = predictions.index(max_value)
    print(label_lines[max_index])