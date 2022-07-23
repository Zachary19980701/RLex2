import rospkg
from sympy import im
import rospy
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import *
import tf2_py as tf2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class rosTopic():
    def __init__(self):
        rospy.init_node('RL', anonymous=True)
        
    def getObs(self):
        print('name')
        
    def quaToOuler(x, y, z, w):
        r = math.atan2(2*(w*x + y*z), 1-2*(x*x + y*y))
        p = math.asin(2*(w*y-z*x))
        y = math.atan2(2*(w*z + x*y), 1-2*(y*y + z*z))
        return y
    
    def oulerToQua(r, p, y):
        sinr = math.sin(r/2)
        sinp = math.sin(p/2)
        siny = math.sin(y/2)
        
        cosr = math.cos(r/2)
        cosp = math.cos(p/2)
        cosy = math.cos(y/2)
        
        w = cosr*cosp*cosy + sinr*sinp*siny
        x = sinr*cosp*cosy - cosr*sinp*siny
        y = cosr*sinp*cosy + sinr*cosp*siny
        z = cosr*cosp*siny - sinr*sinp*cosy
        
        return x, y, z, w
    
    def getPos(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        get_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        modelGet = GetModelStateRequest()
        modelGet.model_name = 'mobile_base'
        objstate = get_state_service(modelGet)
        
        x = objstate.pose.position.x
        y = objstate.pose.position.y
        ox = objstate.pose.orientation.x
        oy= objstate.pose.orientation.y
        oz = objstate.pose.orientation.z
        ow = objstate.pose.orientation.w
        
        Or, Op, Oy = self.quaToOuler(ox, oy, oz, ow)
        
        return x, y, Or, Op, Oy
    
    def getImage(self):
        cvBridge = CvBridge()
        image = rospy.wait_for_message('/camera/rgb/image_raw', Image)
        cvimage = cvBridge.imgmsg_to_cv2(image, 'bgr8')
        
    
