import tensorflow as tf
#youxianshiyong gpu
def test_device(): 
	with tf.device('/cpu:0'):
	    a = tf.constant([[1.0,2.0,3.0,4.0]], name = 'aa')
	    b = tf.reshape(tf.constant(([10.0,20.0,30.0,4.0]), name = 'cb'), [4,1])
	c = tf.matmul(a,b)
	print a
	print b
	with tf.Session(config = tf.ConfigProto(log_device_placement=True)) as sess:
	    print sess.run(c)

def test_graph():
	g = tf.Graph()
	with g.as_default():
		a = 3
		b = 5
		x = tf.add(a,b)
	with tf.Session(graph = g) as sess:
		print sess.run(x)

g1 = tf.get_default_graph()
g2 = tf.Graph()
# add ops to the default graph
with g1.as_default():
	a = tf.constant(3)

# add ops to the user created graph
with g2.as_default():
	b = tf.constant(5)

with tf.Session(graph = g2) as sess:
	print sess.run(b)
