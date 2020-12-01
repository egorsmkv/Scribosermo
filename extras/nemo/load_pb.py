# """ See: https://stackoverflow.com/a/57596363 """
#
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
#
# # ==================================================================================================

path_pb = "/nemo/models/tf.pb"

# # ==================================================================================================
#
# def load_pb(path_to_pb):
#     """Load protobuf as graph, given filepath"""
#     with tf.gfile.GFile(path_to_pb, 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#     with tf.Graph().as_default() as graph:
#         tf.import_graph_def(graph_def, name='')
#         return graph
#
#
# # ==================================================================================================
#
# tf_graph = load_pb(path_pb)
#
# print([n.name for n in tf_graph.as_graph_def().node])


""" See: https://stackoverflow.com/a/58576060 """
import tensorflow as tf
# make a converter object from the saved tensorflow file
converter = tf.lite.TFLiteConverter.from_frozen_graph(path_pb)
# tell converter which type of optimization techniques to use
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# to view the best option for optimization read documentation of tflite about optimization
# go to this link https://www.tensorflow.org/lite/guide/get_started#4_optimize_your_model_optional

# convert the model
tf_lite_model = converter.convert()
# save the converted model
open('models/mnist.tflite', 'wb').write(tf_lite_model)
