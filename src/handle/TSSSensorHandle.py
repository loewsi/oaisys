# blender imports
import bpy

# utility imports
import numpy as np
import csv
import random
import importlib
import os

from src.TSSBase import TSSBase

class TSSSensorHandle(TSSBase):
    """docstring for TSSSensorHandle"""
    def __init__(self):
        super(TSSSensorHandle, self).__init__()
        # class vars ###################################################################################################
        self._sensor_list = []                                          # list of sensors [list]
        self._target_pose_list = []                                     # list of target poses [list]
        self._target_object_active = False                              # flag whether target object is active [bool]
        self._sensor_movement_type = -1                                 # movement type [int]
        self._sensor_pose_list = []                                     # list of sensor poses [list]
        self._pose_index = 0                                            # current pose index [int]
        self._sensor_base = None                                        # sensor base of all sensors [blObject]
        self._sensor_base_constraint = None                             # sensor base constraint object of all sensors 
                                                                        #                                    [blObject]
        self._target_object = None                                      # target object of sensor base [blObject]
        self._hover_base_constraint = None                              # hover constraint for base [blObject]
        self._hover_target_constraint = None                            # hover constraint for target [blObject]
        self._sensor_movement_type_mapping = {                          # mapping to movement functions [dict]
                                                "deterministic": self._deterministic,
                                                "randomEuclidean": self._random_euclidean,
                                                "randomEuclideanTarget": self._random_euclidean_target,
                                                "randomEuclideanTrajectory": self._random_euclidean_trajectory,
                                                "squareFollowEuclideanTrajectory": self._square_follow_euclidean_trajectory,
                                                "straightLine": self._straight_line}
        self._roll_limits = [0,0]
        self._outputPath = None
        ############################################################################################ end of class vars #

        # set correct module type
        self.set_module_type(type=1)
        

    def reset_module(self):
        """ reset all local vars
        Args:
            None
        Returns:
            None
        """

        # reset all sensors ############################################################################################
        for sensor in self._sensor_list:
            # reset sensor
            sensor.reset_module()

            # maybe osbolete in future versions
            del sensor
        ##################################################################################### end of reset all sensors #

        self.reset_base()

        # reset vars ###################################################################################################
        self._sensor_list = []
        self._target_pose_list = []
        self._target_object_active = False
        self._sensor_movement_type = -1
        self._sensor_pose_list = []
        self._pose_index = 0
        self._sensor_base = None
        self._target_object = None
        self._sensor_base_constraint = None
        self._roll_limits = [0,0]
        ############################################################################################ end of reset vars #


    def _create_base_object(self,cfg):
        """ create base object
        Args:
            cfg:            general part of cfg from sensor handle [dict]
        Returns:
            ErrorCode:      error code; 0 if sucessful
        """

        
        # create empty blender object ##################################################################################
        self._sensor_base = bpy.data.objects.new("empty",None)
        self._sensor_base.name = "sensor_base"
        self._sensor_base.empty_display_type = 'PLAIN_AXES'
        self._sensor_base.empty_display_size = 0.5
        bpy.context.scene.collection.objects.link(self._sensor_base)
        ########################################################################### end of create empty blender object #

        # create empty blender object for sensor base to apply constraints #############################################
        self._sensor_base_constraint = bpy.data.objects.new("empty",None)
        self._sensor_base_constraint.name = "dummy_object"
        self._sensor_base_constraint.empty_display_type = 'PLAIN_AXES'
        self._sensor_base_constraint.empty_display_size = 0.5
        bpy.context.scene.collection.objects.link(self._sensor_base_constraint)
        ###################################### end of create empty blender object for sensor base to apply constraints #

        # store roll values ############################################################################################
        if "randomTargetRollRad" in cfg:
            self._roll_limits = cfg["randomTargetRollDeg"]
        else:
            if "randomTargetRollDeg" in self._cfg["GENERAL"]:
                self._roll_limits[0] = (np.pi/180.)*cfg["randomTargetRollDeg"][0]
                self._roll_limits[1] = (np.pi/180.)*cfg["randomTargetRollDeg"][1]
            else:
                self._roll_limits = [0,0]
        ##################################################################################### end of store roll values #
        
        # create movement pattern of sensor base
        self._sensor_pose_list,self._target_pose_list=self._sensor_movement_type_mapping[cfg["sensorMovementType"]](cfg)

        # interpolate pose frames if needed ############################################################################
        if "interpolationMode" in cfg:
            
            # interpolate base poses
            self._sensor_pose_list = self._pose_interpolate(pose_list=self._sensor_pose_list,
                                                            interpolation_mode=cfg["interpolationMode"])

            # interpolate target object poses if exist
            if self._target_pose_list is not None:
                self._target_pose_list = self._pose_interpolate(pose_list=self._target_pose_list,
                                                                interpolation_mode=cfg["interpolationMode"])
        ##################################################################### end of interpolate pose frames if needed #

        # target object creation #######################################################################################
        # check if target object shall be active and create it if needed 
        if self._target_pose_list is not None:

            # enable target object flag
            self._target_object_active = True
            # create target object #####################################################################################
            self._target_object = bpy.data.objects.new("empty",None)
            self._target_object.name = "target_object"
            self._target_object.empty_display_type = 'PLAIN_AXES'
            self._target_object.empty_display_size = 0.5
            self._target_object.rotation_mode = 'QUATERNION'
            bpy.context.scene.collection.objects.link(self._target_object)
            ############################################################################## end of create target object #

            # create hover constraint ##################################################################################
            if "hoverBaseModeEnabled" in cfg:
                if cfg["hoverBaseModeEnabled"]:
                    self._hover_base_constraint = self._sensor_base_constraint.constraints.new(type='SHRINKWRAP')
                    self._hover_base_constraint.shrinkwrap_type = 'NEAREST_SURFACE'
                    self._hover_base_constraint.wrap_mode = 'ABOVE_SURFACE'
                    self._hover_base_constraint.track_axis = 'TRACK_Z'
                    self._hover_base_constraint.distance = float(cfg['hoverBaseDistance'])

            if "hoverTargetModeEnabled" in cfg:
                if cfg["hoverTargetModeEnabled"]:
                    self._hover_target_constraint = self._target_object.constraints.new(type='SHRINKWRAP')
                    self._hover_target_constraint.shrinkwrap_type = 'NEAREST_SURFACE'
                    self._hover_target_constraint.wrap_mode = 'ABOVE_SURFACE'
                    self._hover_target_constraint.distance = float(cfg['hoverTargetDistance'])
            ########################################################################### end of create hover constraint #

            # create target object to sensor base ######################################################################
            self._sensor_base_constraint.constraints.new(type='TRACK_TO')
            self._sensor_base_constraint.constraints["Track To"].target = self._target_object
            self._sensor_base_constraint.constraints["Track To"].track_axis = 'TRACK_NEGATIVE_Z'
            self._sensor_base_constraint.constraints["Track To"].up_axis = 'UP_Y'
            ############################################################### end of create target object to sensor base #
        ####################################################################################### target object creation #
        

    def set_stages(self,stages):
        if self._hover_base_constraint is not None:
            _stage = stages[self._cfg["GENERAL"]["hoverBaseStage"]]
            if _stage is not None:
                self._hover_base_constraint.target = _stage
            else:
                raise("Stage for base hovering mode was not found!")

        if self._hover_target_constraint is not None:
            _stage = stages[self._cfg["GENERAL"]["hoverTargetStage"]]
            if _stage is not None:
                self._hover_target_constraint.target = _stage
            else:
                raise("Stage for target hovering mode was not found!")


    def _create_sensors(self,cfg):
        """ create all sensors defined in cfg
        Args:
            cfg:            sensors part of cfg from sensor handle [dict]
        Returns:
            ErrorCode:      error code; 0 if sucessful
        """

        # go through cfg and add sensors ###############################################################################
        for sensor in cfg:
            try:
                # import module and create class #######################################################################
                _module_name = "src.sensors.sensors." + sensor["type"]
                _module = importlib.import_module(_module_name)
                _class = getattr(_module, sensor["type"])
                _sensor = _class()
                ################################################################ end of import module and create class #

                # update sensor cfg
                _sensor.update_cfg(cfg=sensor["sensorParams"])
                
                # create sensor
                _sensor.create()
                
                # add sensor to list
                self._sensor_list.append(_sensor)

            except ImportError:
                # manage import error
                raise Exception("Cannot add sensor")
                return -1

        return 0
        ######################################################################## end of go through cfg and add sensors #


    def _attach_sensors_to_base(self):
        """ parent sensors to base
            Better option would be to use child of modifier, however CHILD_OF modifier has a bug and creates an offset;
            offset can be delete by setting inverse and reset of the modifier
            _child_constraint = _sensor.constraints.new(type = 'CHILD_OF')
            _child_constraint.target = self._sensor_base

            using parenting instead; not so much flexability, but gets the job done

        Args:
            None
        Returns:
            None
        """

        # go through all sensors and attach to base ####################################################################
        for sensor_obj in self._sensor_list:

            # get sensor handle
            _sensor = sensor_obj.get_sensor()
            
            # set parent
            _sensor.parent = self._sensor_base
        ############################################################# end of go through all sensors and attach to base #


    def _read_pose_from_CSV(self,csv_file_path):

        # def local var
        _pose_list = []

        # go through csv file and get data #############################################################################
        with open(csv_file_path) as csv_file:
            # define csv reader
            _csv_reader = csv.reader(csv_file, delimiter=',')

            # local csv line counter
            _line_count = 0

            # iterate through csv lines ################################################################################
            for row in _csv_reader:

                # def local pose var
                _pose = np.zeros((8))

                # read frame ID
                _pose[0] = int(row[0])

                # read position
                _pose[1] = float(row[1])  # posX
                _pose[2] = float(row[2])  # posY
                _pose[3] = float(row[3])  # posZ
                
                # read quaternion
                _pose[4] = float(row[4])  # quatW
                _pose[5] = float(row[5])  # quatX
                _pose[6] = float(row[6])  # quatY
                _pose[7] = float(row[7])  # quatZ

                # append retrieved pose sample to list
                _pose_list.append(_pose)
            ######################################################################### end of iterate through csv lines #
        ###################################################################### end of go through csv file and get data #

        # return pose list
        return _pose_list


    def _deterministic(self,cfg):
        """ deterministic movement option; read movement from csv file
        Args:
            cfg:                        general part of cfg from sensor handle [dict] 
        Returns:
            pose_list:                  list of random poses [list]
            target_pose_list:           No target pose list, therefore return None [None]
        """

        # read pose from csv file
        _pose_list = self._read_pose_from_CSV(cfg["sensorMovementPose"])

        # interpolate between poses
        _pose_list = self._pose_interpolate(pose_list=_pose_list, interpolation_mode=cfg["interpolationMode"])

        # return movement lists
        return _pose_list, None


    def _random_euclidean(self, cfg):
        """ random euclidean movement option;  create random pose within box
        Args:
            cfg:                        general part of cfg from sensor handle [dict] 
        Returns:
            pose_list:                  list of random poses [list]
            target_pose_list:           No target pose list, therefore return None [None]
        """

        # create random poses
        _pose_list = self._create_euclidean_random_sensor_movements(min_pos=cfg["randomEuclideanPosMin"],
                                                            max_pos=cfg["randomEuclideanPosMax"],
                                                            num_samples=cfg["numSamples"],
                                                            min_euler=cfg["randomEuclideanEulerMin"],
                                                            max_euler=cfg["randomEuclideanEulerMax"])

        # return movement lists
        return _pose_list, None


    def _random_euclidean_target(self, cfg):
        """ random euclidean target movement option;  create random pose for base and target object within box
        Args:
            cfg:                        general part of cfg from sensor handle [dict] 
        Returns:
            pose_list:                  list of random poses [list]
            target_pose_list:           list of target poses [list]
        """

        # create random poses for base and target
        _pose_list, _target_pose_list =  self._create_euclidean_random_sensor_movements(\
                                                                        min_pos=cfg["randomEuclideanPosMin"],
                                                                        max_pos=cfg["randomEuclideanPosMax"],
                                                                        num_samples=cfg["numSamples"],
                                                                        target_active=True,
                                                                        min_target_pos=cfg["randomTargetPosMin"],
                                                                        max_target_pos=cfg["randomTargetPosMax"],
                                                                        min_target_euler=cfg["randomTargetEulerMin"],
                                                                        max_target_euler=cfg["randomTargetEulerMax"])

        # return movement lists
        return _pose_list, _target_pose_list


    def _random_euclidean_trajectory(self, cfg):
        """ random euclidean trajectory;  create pose on trajectory facing target
        Args:
            cfg:                        general part of cfg from sensor handle [dict] 
        Returns:
            pose_list:                  list of trajectory poses [list]
            target_pose_list:           list of target poses [list]
        """

        # create random poses for base and target
        _pose_list, _target_pose_list =  self._create_euclidean_random_sensor_trajectory(\
                                                                        min_pos=cfg["randomEuclideanPosMin"],
                                                                        max_pos=cfg["randomEuclideanPosMax"],
                                                                        num_samples=cfg["numSamples"],
                                                                        min_target_pos=cfg["randomTargetPosMin"],
                                                                        max_target_pos=cfg["randomTargetPosMax"],
                                                                        min_target_euler=cfg["randomTargetEulerMin"],
                                                                        max_target_euler=cfg["randomTargetEulerMax"])

        # return movement lists
        return _pose_list, _target_pose_list


    def _square_follow_euclidean_trajectory(self, cfg):
        """ random euclidean trajectory;  create pose on trajectory facing target
        Args:
            cfg:                        general part of cfg from sensor handle [dict] 
        Returns:
            pose_list:                  list of trajectory poses [list]
            target_pose_list:           list of target poses [list]
        """

        # create random poses for base and target
        _pose_list, _target_pose_list =  self._create_square_follow_euclidean_sensor_trajectory(\
                                                                        min_pos=cfg["randomEuclideanPosMin"],
                                                                        max_pos=cfg["randomEuclideanPosMax"],
                                                                        num_samples=cfg["numSamples"],
                                                                        min_target_pos=cfg["randomTargetPosMin"],
                                                                        max_target_pos=cfg["randomTargetPosMax"],
                                                                        min_target_euler=cfg["randomTargetEulerMin"],
                                                                        max_target_euler=cfg["randomTargetEulerMax"])

        # return movement lists
        return _pose_list, _target_pose_list

    def _straight_line(self, cfg):
        """ straight line trajectory;  create pose on trajectory facing target straight ahead
        Args:
            cfg:                        general part of cfg from sensor handle [dict] 
        Returns:
            pose_list:                  list of trajectory poses [list]
            target_pose_list:           list of target poses [list]
        """

        # create random poses for base and target
        _pose_list, _target_pose_list =  self._create_straight_line_sensor_trajectory(\
                                                                        min_pos=cfg["randomEuclideanPosMin"],
                                                                        max_pos=cfg["randomEuclideanPosMax"],
                                                                        num_samples=cfg["numSamples"])

        # return movement lists
        return _pose_list, _target_pose_list


    def _euler_to_quaternion(self, angles):

        roll = angles[0]
        pitch = angles[1]
        yaw = angles[2]

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, qw]


    def _create_random_pose_sample(self, min_pos, max_pos, min_euler, max_euler):
        """ create random pose
        Args:
            min_pos:        boundary vlaues for minimum position [x,y,z] [numpy]
            max_pos:        boundary vlaues for maximum position [x,y,z] [numpy]
            min_euler:       boundary vlaues for minimum quaternion [angle_x,angle_y,angle_z] [numpy]
            max_euler:       boundary vlaues for maximum quaternion [angle_x,angle_y,angle_z] [numpy]
        Returns:
            _pose:          6D random pose vector (7 dim) [numpy] 
        """

        # def local var
        _pose = np.zeros((7))
        _euler = np.zeros((3))

        # random positions
        _pose[0] = random.uniform(min_pos[0], max_pos[0])
        _pose[1] = random.uniform(min_pos[1], max_pos[1])
        _pose[2] = random.uniform(min_pos[2], max_pos[2])

        # random angles
        _euler[0] = random.uniform(min_euler[0], max_euler[0])
        _euler[1] = random.uniform(min_euler[1], max_euler[1])
        _euler[2] = random.uniform(min_euler[2], max_euler[2])

        # convert to quaternion
        _pose[3:7] = self._euler_to_quaternion(angles=_euler)

        # return random pose sample
        return _pose

    def _create_square_pose_sample(self, min_pos, max_pos, min_euler, max_euler, index, max_index):
        """ create random pose
        Args:
            min_pos:        boundary vlaues for minimum position [x,y,z] [numpy]
            max_pos:        boundary vlaues for maximum position [x,y,z] [numpy]
            min_euler:       boundary vlaues for minimum quaternion [angle_x,angle_y,angle_z] [numpy]
            max_euler:       boundary vlaues for maximum quaternion [angle_x,angle_y,angle_z] [numpy]
            index:          current index of pose on square
            max_index:      total number of poses to be sampled on square
        Returns:
            _pose:          6D random pose vector (7 dim) [numpy] 
        """

        # def local var
        _pose = np.zeros((7))
        _euler = np.zeros((3))

        # get positions
        _square_position = index/max_index*4
        if int(_square_position) == 0:
            _pose[0] = min_pos[0] + _square_position * (max_pos[0]-min_pos[0])
            _pose[1] = min_pos[1]
        if int(_square_position) == 1:
            _pose[0] = max_pos[0]
            _pose[1] = min_pos[1] + (_square_position - 1) * (max_pos[1] - min_pos[1])
        if int(_square_position) == 2:
            _pose[0] = max_pos[0] - (_square_position - 2) * (max_pos[0]-min_pos[0])
            _pose[1] = max_pos[1]
        if int(_square_position) == 3:
            _pose[0] = min_pos[0]
            _pose[1] = max_pos[1] - (_square_position - 3) * (max_pos[1] - min_pos[1])
        _pose[2] = random.uniform(min_pos[2], max_pos[2])

        # random angles
        _euler[0] = random.uniform(min_euler[0], max_euler[0])
        _euler[1] = random.uniform(min_euler[1], max_euler[1])
        _euler[2] = random.uniform(min_euler[2], max_euler[2])

        # convert to quaternion
        _pose[3:7] = self._euler_to_quaternion(angles=_euler)

        # return random pose sample
        return _pose

    def _create_trajectory_pose_sample(  self,
                                    last_pose, last_target,
                                    min_step_distance, max_step_distance,
                                    min_pos, max_pos,
                                    min_euler, max_euler):
        print("create pose sample")
        """ create random pose
        Args:
            last_pose:          last position [x,y,z] [numpy]
            last_target:        last target position [x,y,z] [numpy]
            min_step distance:  boundary vlaues for minimum step distance   
            max_step distance:  boundary vlaues for maximum step distance  
            min_pos:            boundary vlaues for minimum position [x,y,z] [numpy]
            max_pos:            boundary vlaues for maximum position [x,y,z] [numpy]
            min_euler:          boundary vlaues for minimum quaternion [angle_x,angle_y,angle_z] [numpy]
            max_euler:          boundary vlaues for maximum quaternion [angle_x,angle_y,angle_z] [numpy]
        Returns:
            _pose:          6D random pose vector (7 dim) [numpy] 
        """
        
        # def local var
        _pose = np.zeros((7))
        _euler = np.zeros((3))
        _direction = np.zeros((3))

        # direction of next position
        _direction = last_target - last_pose
        _direction = _direction / np.linalg.norm(_direction)

        # next position
        _pose[0:3] = last_pose + _direction * random.uniform(min_step_distance, max_step_distance)
        _pose[0] = max(_pose[0], min_pos[0])
        _pose[0] = min(_pose[0], max_pos[0])
        _pose[1] = max(_pose[1], min_pos[1])
        _pose[1] = min(_pose[1], max_pos[1])
        _pose[2] = max(_pose[2], min_pos[2])
        _pose[2] = min(_pose[2], max_pos[2])

        # random angles
        _euler[0] = random.uniform(min_euler[0], max_euler[0])
        _euler[1] = random.uniform(min_euler[1], max_euler[1])
        _euler[2] = random.uniform(min_euler[2], max_euler[2])

        # convert to quaternion
        _pose[3:7] = self._euler_to_quaternion(angles=_euler)

        # return random pose sample
        return _pose



    def _create_trajectory_target_sample(  self,
                                    last_target,
                                    min_step_distance, max_step_distance,
                                    min_pos, max_pos,
                                    min_euler, max_euler,
                                    second_last_target=None,
                                    direction_noise=0.3):
        print("create target sample")
        """ create random pose
        Args:
            last_target:                    last target position [x,y,z] [numpy]
            second_last_target (optional):  second last target position [x,y,z] [numpy]
            min_step distance:              boundary vlaues for minimum step distance   
            max_step distance:              boundary vlaues for maximum step distance  
            min_pos:                        boundary vlaues for minimum position [x,y,z] [numpy]
            max_pos:                        boundary vlaues for maximum position [x,y,z] [numpy]
            min_euler:                      boundary vlaues for minimum quaternion [angle_x,angle_y,angle_z] [numpy]
            max_euler:                      boundary vlaues for maximum quaternion [angle_x,angle_y,angle_z] [numpy]
        direction_noise (optional):         std of normal noise on direction, default 0.3
        Returns:
            _pose:          6D random pose vector (7 dim) [numpy] 
        """

        # def local var
        _pose = np.zeros((7))
        _euler = np.zeros((3))
        _direction = np.zeros((3))

        # next position when second last target is given
        if second_last_target is not None:
            # direction of next position
            _direction = last_target - second_last_target
            _direction[0] += np.random.normal(scale=direction_noise)
            _direction[1] += np.random.normal(scale=direction_noise)
            _direction[2] += np.random.normal(scale=direction_noise)
            _direction = _direction / np.linalg.norm(_direction)

            # next position
            _pose[0:3] = last_target + _direction * random.uniform(min_step_distance, max_step_distance)
            _pose[0] = max(_pose[0], min_pos[0])
            _pose[0] = min(_pose[0], max_pos[0])
            _pose[1] = max(_pose[1], min_pos[1])
            _pose[1] = min(_pose[1], max_pos[1])
            _pose[2] = max(_pose[2], min_pos[2])
            _pose[2] = min(_pose[2], max_pos[2])

        # next position in random direction
        else:
            # direction of next position
            _direction[0] = np.random.normal()
            _direction[1] = np.random.normal()
            _direction[2] = np.random.normal()
            _direction = _direction / np.linalg.norm(_direction)

            # next position
            _pose[0:3] = last_target + _direction * random.uniform(min_step_distance, max_step_distance)
            _pose[0] = max(_pose[0], min_pos[0])
            _pose[0] = min(_pose[0], max_pos[0])
            _pose[1] = max(_pose[1], min_pos[1])
            _pose[1] = min(_pose[1], max_pos[1])
            _pose[2] = max(_pose[2], min_pos[2])
            _pose[2] = min(_pose[2], max_pos[2])

        # random angles
        _euler[0] = random.uniform(min_euler[0], max_euler[0])
        _euler[1] = random.uniform(min_euler[1], max_euler[1])
        _euler[2] = random.uniform(min_euler[2], max_euler[2])

        # convert to quaternion
        _pose[3:7] = self._euler_to_quaternion(angles=_euler)

        # return random pose sample
        return _pose


    def _create_euclidean_random_sensor_movements(  self,
                                                min_pos, max_pos,
                                                num_samples,
                                                min_euler=[0,0,0], max_euler=[0,0,0],
                                                target_active=False,
                                                min_target_pos=[0,0,0], max_target_pos=[0,0,0],
                                                min_target_euler=[0,0,0], max_target_euler=[0,0,0]):
        """ create list of random poses, with or without random target pose list
        Args:
            min_pos:                            boundary values for minimum position [x,y,z] [numpy]
            max_pos:                            boundary values for maximum position [x,y,z] [numpy]
            num_samples:                        number of pose samples [uint]
            min_euler (optional):               boundary values for minimum quaternion [angle_x,angle_y,angle_z] [numpy]
            max_euler (optional):               boundary values for maximum quaternion [angle_x,angle_y,angle_z] [numpy]
            target_active (optional):           should target be calculated as well [boolean] 
            min_target_pos:                     boundary values for minimum target position [x,y,z] [numpy]
            max_target_pos:                     boundary values for maximum target position [x,y,z] [numpy]
            min_target_euler (optional):        boundary values for minimum target quaternion [angle_x,angle_y,angle_z] 
                                                                                                                [numpy]
            max_target_euler (optional):        boundary values for maximum target quaternion [angle_x,angle_y,angle_z] 
                                                                                                                [numpy]
        Returns:
            _pose, _target_pose (optional):     return random list of poses [numpy]
        """

        # def of local vars ############################################################################################
        _pose_list = []
        _target_pose_list = []
        ##################################################################################### end of def of local vars #

        # create num_samples samples of random poses ###################################################################
        for pose_index in range(0,num_samples):
            # create var
            _pose = np.zeros((8))

            # frame ID
            _pose[0] = pose_index + 1

            # create random pose sample for base
            _pose[1:] = self._create_random_pose_sample(min_pos=min_pos,
                                                        max_pos=max_pos,
                                                        min_euler=min_euler,
                                                        max_euler=max_euler)

            # add pose sample to list
            _pose_list.append(_pose)

            # create random target poses, if requested
            if target_active:
                # create var
                _target_pose = np.zeros((8))

                # frame ID
                _target_pose[0] = pose_index + 1

                # create random pose sample for target
                _target_pose[1:] = self._create_random_pose_sample( min_pos=min_target_pos,
                                                                    max_pos=max_target_pos,
                                                                    min_euler=min_target_euler,
                                                                    max_euler=max_target_euler)

                # set offset
                _target_pose[1:4] += _pose[1:4]

                # add pose sample to list
                _target_pose_list.append(_target_pose)
        ############################################################ end of create num_samples samples of random poses #

        # return requested values
        if target_active:
            return _pose_list, _target_pose_list
        else:
            return _pose_list


    def _create_euclidean_random_sensor_trajectory(  self,
                                                min_pos, max_pos,
                                                num_samples,
                                                min_euler=[0,0,0], max_euler=[0,0,0],
                                                min_target_pos=[0,0,0], max_target_pos=[0,0,0],
                                                min_target_euler=[0,0,0], max_target_euler=[0,0,0],
                                                min_step_distance=1., max_step_distance=2.):
        """ create list of poses on a trajectory
        Args:
            min_pos:                            boundary values for minimum position [x,y,z] [numpy]
            max_pos:                            boundary values for maximum position [x,y,z] [numpy]
            num_samples:                        number of pose samples [uint]
            min_euler (optional):               boundary values for minimum quaternion [angle_x,angle_y,angle_z] [numpy]
            max_euler (optional):               boundary values for maximum quaternion [angle_x,angle_y,angle_z] [numpy]
            min_target_pos:                     boundary values for minimum target position [x,y,z] [numpy]
            max_target_pos:                     boundary values for maximum target position [x,y,z] [numpy]
            min_target_euler (optional):        boundary values for minimum target quaternion [angle_x,angle_y,angle_z] 
                                                                                                                [numpy]
            max_target_euler (optional):        boundary values for maximum target quaternion [angle_x,angle_y,angle_z] 
                                                                                                                [numpy]
        Returns:
            _pose, _target_pose (optional):     return random list of poses [numpy]
        """

        # def of local vars ############################################################################################
        _pose_list = []
        _target_pose_list = []
        ##################################################################################### end of def of local vars #

        # create num_samples samples of random poses ###################################################################
        for pose_index in range(0,num_samples):
            # create var
            _pose = np.zeros((8))

            # frame ID
            _pose[0] = pose_index + 1

            # create random start pose sample for base 
            if not pose_index:
                _pose[1:] = self._create_random_pose_sample(min_pos=np.array(min_pos)/(max_step_distance*num_samples),
                                                            max_pos=np.array(max_pos)/(max_step_distance*num_samples),
                                                            min_euler=min_euler,
                                                            max_euler=max_euler)
            # create next pose headed to target
            else:
                _pose[1:] = self._create_trajectory_pose_sample(last_pose=_pose_list[-1][1:4],
                                                            last_target=_target_pose_list[-1][1:4],
                                                            min_step_distance=min_step_distance,
                                                            max_step_distance=max_step_distance,
                                                            min_pos=min_pos,
                                                            max_pos=max_pos,
                                                            min_euler=min_euler,
                                                            max_euler=max_euler)
            # add pose sample to list
            _pose_list.append(_pose)

            # create var
            _target_pose = np.zeros((8))

            # frame ID
            _target_pose[0] = pose_index + 1

            # create random start pose sample for target
            if not pose_index:
                _target_pose[1:] = self._create_random_pose_sample( min_pos=min_target_pos,
                                                                        max_pos=max_target_pos,
                                                                        min_euler=min_target_euler,
                                                                        max_euler=max_target_euler)
            # create second pose for target in random direction
            elif pose_index == 1:
                _target_pose[1:] = self._create_trajectory_target_sample(last_target=_target_pose_list[-1][1:4],
                                                            second_last_target=None,
                                                            min_step_distance=min_step_distance,
                                                            max_step_distance=max_step_distance,
                                                            min_pos=min_target_pos,
                                                            max_pos=max_target_pos,
                                                            min_euler=min_target_euler,
                                                            max_euler=max_target_euler)

            # create next pose for target
            else:
                _target_pose[1:] = self._create_trajectory_target_sample(last_target=_target_pose_list[-1][1:4],
                                                            second_last_target=_target_pose_list[-2][1:4],
                                                            min_step_distance=max_step_distance,
                                                            max_step_distance=max_step_distance,
                                                            min_pos=min_target_pos,
                                                            max_pos=max_target_pos,
                                                            min_euler=min_target_euler,
                                                            max_euler=max_target_euler)

            # set offset
            _target_pose[1:4] += _pose[1:4]

            # add pose sample to list
            _target_pose_list.append(_target_pose)
        ############################################################ end of create num_samples samples of random poses #

        # return requested values
        return _pose_list, _target_pose_list


    def _create_square_follow_euclidean_sensor_trajectory(  self,
                                                min_pos, max_pos,
                                                num_samples,
                                                min_euler=[0,0,0], max_euler=[0,0,0],
                                                min_target_pos=[0,0,0], max_target_pos=[0,0,0],
                                                min_target_euler=[0,0,0], max_target_euler=[0,0,0],
                                                min_step_distance=1., max_step_distance=2.):
        """ create list of poses on a trajectory following a target trajectory on a square
        Args:
            min_pos:                            boundary values for minimum position [x,y,z] [numpy]
            max_pos:                            boundary values for maximum position [x,y,z] [numpy]
            num_samples:                        number of pose samples [uint]
            min_euler (optional):               boundary values for minimum quaternion [angle_x,angle_y,angle_z] [numpy]
            max_euler (optional):               boundary values for maximum quaternion [angle_x,angle_y,angle_z] [numpy]
            min_target_pos:                     boundary values for minimum target position [x,y,z] [numpy]
            max_target_pos:                     boundary values for maximum target position [x,y,z] [numpy]
            min_target_euler (optional):        boundary values for minimum target quaternion [angle_x,angle_y,angle_z] 
                                                                                                                [numpy]
            max_target_euler (optional):        boundary values for maximum target quaternion [angle_x,angle_y,angle_z] 
                                                                                                                [numpy]
        Returns:
            _pose, _target_pose (optional):     return random list of poses [numpy]
        """

        # def of local vars ############################################################################################
        _pose_list = []
        _target_pose_list = []
        ##################################################################################### end of def of local vars #

        # create num_samples samples of random poses ###################################################################
        for pose_index in range(0,num_samples):
            # create var
            _pose = np.zeros((8))

            # frame ID
            _pose[0] = pose_index + 1

            # create random start pose sample for base 
            if not pose_index:
                _pose[1:] = self._create_random_pose_sample(min_pos=[0,0,0],
                                                            max_pos=[1,1,1],
                                                            min_euler=min_euler,
                                                            max_euler=max_euler)
            # create next pose headed to target
            else:
                _pose[1:] = self._create_trajectory_pose_sample(last_pose=_pose_list[-1][1:4],
                                                            last_target=_target_pose_list[-1][1:4],
                                                            min_step_distance=min_step_distance,
                                                            max_step_distance=max_step_distance,
                                                            min_pos=min_pos,
                                                            max_pos=max_pos,
                                                            min_euler=min_euler,
                                                            max_euler=max_euler)
            # add pose sample to list
            _pose_list.append(_pose)

            # create var
            _target_pose = np.zeros((8))

            # frame ID
            _target_pose[0] = pose_index + 1

            # create pose sample for target on square
            _target_pose[1:] = self._create_square_pose_sample( min_pos=min_target_pos,
                                                                        max_pos=max_target_pos,
                                                                        min_euler=min_target_euler,
                                                                        max_euler=max_target_euler,
                                                                        index=pose_index,
                                                                        max_index=num_samples)

            # set offset
            _target_pose[1:4] += _pose[1:4]

            # add pose sample to list
            _target_pose_list.append(_target_pose)
        ############################################################ end of create num_samples samples of random poses #

        # return requested values
        return _pose_list, _target_pose_list


    def _create_straight_line_sensor_trajectory(  self,
                                                min_pos, max_pos,
                                                num_samples):
        """ create list of poses on a straight trajectory
        Args:
            min_pos:                            boundary values for minimum position [x,y,z] [numpy]
            max_pos:                            boundary values for maximum position [x,y,z] [numpy]
            num_samples:                        number of pose samples [uint]
        Returns:
            _pose, _target_pose (optional):     return list of poses [numpy]
        """

        # def of local vars ############################################################################################
        _pose_list = []
        _target_pose_list = []
        ##################################################################################### end of def of local vars #

        # create samples ###################################################################
        for pose_index in range(0,num_samples):
            # create var
            _pose = np.zeros((8))

            # frame ID
            _pose[0] = pose_index + 1

            # create pose sample for base 
            _euler = np.zeros((3))

            # interpolate positions
            _pose[1] = min_pos[0] + pose_index * (max_pos[0] - min_pos[0])/num_samples
            _pose[2] = min_pos[1] + pose_index * (max_pos[1] - min_pos[1])/num_samples
            _pose[3] = min_pos[2] + pose_index * (max_pos[2] - min_pos[2])/num_samples

            # convert to quaternion
            _pose[4:8] = self._euler_to_quaternion(angles=_euler)
                # add pose sample to list
            _pose_list.append(_pose)

            # create var
            _target_pose = np.zeros((8))

            # frame ID
            _target_pose[0] = pose_index + 1

            # create pose sample for target
            _target_pose[1] = max_pos[0] - min_pos[0] + _pose[1]
            _target_pose[2] = max_pos[1] - min_pos[1] + _pose[2]
            _target_pose[3] = max_pos[2] - min_pos[2] + _pose[3]

            # convert to quaternion
            _target_pose[4:8] = self._euler_to_quaternion(angles=_euler)

            # add pose sample to list
            _target_pose_list.append(_target_pose)
        ############################################################ end of create samples#

        # return requested values
        return _pose_list, _target_pose_list



    def _pose_interpolate(self, pose_list, interpolation_mode="static"):
        """ interpolate missing poses in pose_list with a variety of options
        Args:
            pose_list:              list of poses to be interpolated [list]
            interpolation_mode:     mode of interpolation [string]
                                        possible modes: "none":     return same pose_list as inputed
                                                        "static":   copy value from previous pose in gaps
                                                        "step":     copy value from previous pose till the half, the
                                                                    other half copy pose of next pose in it
                                                        "linear":   linear interpolation of two poses
        Returns:
            _pose_list:             interpolated pose list [list]
        """

        # def local var
        _pose_list = None

        # use interpolation mode as requested ##########################################################################
        if "none" == interpolation_mode:
            _pose_list = pose_list

        if "static" == interpolation_mode:
            raise Exception("Not implemented yet!")

        if "step" == interpolation_mode:
            raise Exception("Not implemented yet!") 

        if "linear" == interpolation_mode:
            raise Exception("Not implemented yet!")
        ################################################################### end of use interpolation mode as requested #

        # return interpolated pose list
        return _pose_list


    def set_output_folder(self,output_folder_path):
        self._outputPath = output_folder_path
    
    def create(self):
        """ create function
        Args:
            None
        Returns:
            ErrorCode:      error code; 0 if sucessful
        """

        # reset all local vars
        #self._reset()

        # create sensor base
        self._create_base_object(self._cfg["GENERAL"])

        # create sensors
        self._create_sensors(self._cfg["SENSORS"])

        # attach sensors to base
        self._attach_sensors_to_base()


    def step(self, keyframe, sample):
        """ overwrite step function
        Args:
            keyframe:       current frame number; if value > -1, this should enable also the setting of a keyframe [int]
        Returns:
            None
        """

        # increase pose counter
        self._pose_index += 1

        # move base pose one position forward ##########################################################################
        # get pose at pose_index
        _pose = self._sensor_pose_list[self._pose_index-1]
        # update locaiton and rotation of base sensor
        self._sensor_base_constraint.location = (_pose[1],_pose[2],_pose[3])
        self._sensor_base_constraint.rotation_quaternion = (_pose[4],_pose[5],_pose[6],_pose[7])
        ################################################################### end of move base pose one position forward #

        # update locaiton and rotation of target sensor ################################################################
        if self._target_object_active:
            # move target pose one position forward
            _target_object_pose = self._target_pose_list[self._pose_index-1]

            # set position of target object
            self._target_object.location = (_target_object_pose[1],
                                            _target_object_pose[2],
                                            _target_object_pose[3])

            # set quaternion of target object
            self._target_object.rotation_quaternion = ( _target_object_pose[4],
                                                        _target_object_pose[5],
                                                        _target_object_pose[6],
                                                        _target_object_pose[7])
        ######################################################### end of update locaiton and rotation of target sensor #

        # calc hover base z noise ######################################################################################
        if self._hover_target_constraint is not None:
            if "hoverBaseDistanceNoise" in self._cfg["GENERAL"]:
                _noise_value = float(self._cfg["GENERAL"]["hoverBaseDistanceNoise"])
                _distance_random = random.uniform(0, _noise_value)
                _distance_offset_value = self._cfg["GENERAL"]["hoverBaseDistance"] - _distance_random/2.0
                self._hover_base_constraint.distance = _distance_offset_value
        ############################################################################### end of calc hover base z noise #
        
        # set sensor base ##############################################################################################
        bpy.context.view_layer.update()
        self._sensor_base.matrix_world = self._sensor_base_constraint.matrix_world
        ####################################################################################### end of set sensor base #

        # add roll angle ###############################################################################################
        _roll_angle = random.uniform(self._roll_limits[0], self._roll_limits[1])
        self._sensor_base.rotation_euler.rotate_axis("Z", _roll_angle)
        bpy.context.view_layer.update()
        ######################################################################################## end of add roll angle #

        # go through all sensors and execute step function
        for sensor_obj in self._sensor_list:
            sensor_obj.step(keyframe=keyframe)

        # set keyframes if requested ###################################################################################
        if keyframe >= 0:
            # set keyframe for location and rotation of base sensor
            self._sensor_base.keyframe_insert('location', frame=keyframe)
            self._sensor_base.keyframe_insert('rotation_euler', frame=keyframe)
            self._sensor_base_constraint.keyframe_insert('location', frame=keyframe)
            self._sensor_base_constraint.keyframe_insert('rotation_euler', frame=keyframe)

            # set keyframe for hover constraint
            if self._hover_base_constraint is not None:
                self._hover_base_constraint.keyframe_insert('distance', frame=keyframe)

            # set interpolation of all keyframes #######################################################################
            _fcurves = self._sensor_base.animation_data.action.fcurves
            for fcurve in _fcurves:
                for kf in fcurve.keyframe_points:
                    kf.interpolation = 'CONSTANT'

            _fcurves = self._sensor_base_constraint.animation_data.action.fcurves
            for fcurve in _fcurves:
                for kf in fcurve.keyframe_points:
                    kf.interpolation = 'CONSTANT'
            # set interpolation of all keyframes########################################################################
            
            # set keyframe for target ##################################################################################
            if self._target_object_active:
                self._target_object.keyframe_insert('location', frame=keyframe)
                self._target_object.keyframe_insert('rotation_quaternion', frame=keyframe)

                # set interpolation of all keyframes ###################################################################
                _fcurves = self._target_object.animation_data.action.fcurves
                for fcurve in _fcurves:
                    for kf in fcurve.keyframe_points:
                        kf.interpolation = 'CONSTANT'
                # set interpolation of all keyframes####################################################################
            ########################################################################### end of set keyframe for target #
        ############################################################################ end of set keyframes if requested #
        
        # save pose to file
        _directory = os.path.join(self._outputPath, 'sensor_1')
        if not os.path.exists(_directory):
            os.makedirs(_directory)
        with open(os.path.join(_directory, f'{sample+1:04d}sensor_1_poses.txt'), 'w') as f:
            f.write(f'{self._sensor_base.matrix_world.to_translation().x}\n')
            f.write(f'{self._sensor_base.matrix_world.to_translation().y}\n')
            f.write(f'{self._sensor_base.matrix_world.to_translation().z}\n')
            f.write(f'{self._sensor_base.matrix_world.to_quaternion().x}\n')
            f.write(f'{self._sensor_base.matrix_world.to_quaternion().y}\n')
            f.write(f'{self._sensor_base.matrix_world.to_quaternion().z}\n')
            f.write(f'{self._sensor_base.matrix_world.to_quaternion().w}\n')



    def get_sensor_list(self):
        """ return self._sensor_list
        Args:
            None
        Returns:
            return self._sensor_list
        """

        return self._sensor_list