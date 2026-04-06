import numpy as np


class Camera:
    def __init__(self):
        self.pose = None
        self.proj = None


    



    def get_w2c_matrix(self, radius, azimuth_deg, elevation_deg):
        az = np.deg2rad(azimuth_deg)
        el = np.deg2rad(elevation_deg)
        
        # 1. 你的cam_pos定义
        cam_pos = np.array([
            radius * np.cos(az) * np.cos(el),
            radius * np.sin(az) * np.cos(el),
            radius * np.sin(el)
        ])
        target = np.array([0, 0, 0])
        world_up = np.array([0, 0, 1])

        # 2. 前向量（摄像机坐标系z轴，指向目标）
        forward = (target - cam_pos)
        forward = forward / np.linalg.norm(forward)
        # 3. 右向量（摄像机坐标系x轴）
        right = -np.cross(world_up, forward)
        right = right / np.linalg.norm(right)
        # 4. 上向量（摄像机坐标系y轴）
        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)

        # 5. 组装旋转矩阵
        R = np.stack([right, up, forward], axis=1)  # 列向量按 x, y, z

        # 6. 组装4x4 W2C
        W2C = np.eye(4)
        W2C[:3, :3] = R.T  # 行向量按 x, y, z
        W2C[:3, 3] = -R.T @ cam_pos
        # W2C[0,:3] *= -1  # 翻转x轴以适应OpenCV的坐标系
        # W2C[1,:3] *= -1  # 翻转y轴以适应OpenCV的坐标系
        return W2C

    def normalize(self, v):
        return v / np.linalg.norm(v)

    def look_at(self, camera_position, target_position):
        view_direction = self.normalize(target_position - camera_position)
        temp_up = np.array([0.0, 1.0, 0.0])  # 假设世界的 up 方向是 Y 轴正向
        
        # 检查 view_direction 和 temp_up 是否平行
        if np.dot(view_direction, temp_up) > 0.99:
            temp_up = np.array([1.0, 0.0, 0.0])  # 如果平行，换一个方向
        elif np.dot(view_direction, temp_up) < -0.99:
            temp_up = np.array([-1.0, 0.0, 0.0])
        
        right = self.normalize(np.cross(view_direction, temp_up))
        up = -self.normalize(np.cross(right, view_direction))
        
        # 构建视图矩阵
        R = np.stack([right, up, -view_direction], axis=0)
        T = -np.dot(R, camera_position)
        view_matrix = np.eye(4)
        view_matrix[:3, :3] = R
        view_matrix[:3, 3] = T
        
        return view_matrix


        # view_direction = self.normalize(target_position - camera_position)
        # temp_up = np.array([0.0, 1.0, 0.0])  # 世界 up 方向为 Y+
        
        # # 检查 view_direction 和 temp_up 是否平行
        # if np.abs(np.dot(view_direction, temp_up)) > 0.99:
        #     temp_up = np.array([1.0, 0.0, 0.0])

        
        # # 和你的例子一致：X轴=up, Y轴=-view, Z轴=-right
        # up = self.normalize(temp_up)
        # right = self.normalize(np.cross(up, view_direction))
        # new_up = np.cross(view_direction, right)

        # R = np.stack([up, -view_direction, -right], axis=0)
        # T = -np.dot(R, camera_position)
        # view_matrix = np.eye(4)
        # view_matrix[:3, :3] = R
        # view_matrix[:3, 3] = T
        # return view_matrix


    def perspective(self, fovy, aspect, near, far):
        f = 1.0 / np.tan(fovy / 2)
        proj = np.zeros((4,4), dtype=np.float32)
        proj[0,0] = f / aspect
        proj[1,1] = f
        proj[2,2] = (far + near) / (near - far)
        proj[2,3] = (2 * far * near) / (near - far)
        proj[3,2] = -1
        return proj


    def orthographic(self, left, right, bottom, top, near, far):
        proj = np.zeros((4,4), dtype=np.float32)
        proj[0,0] = 2.0 / (right - left)
        proj[1,1] = 2.0 / (top - bottom)
        proj[2,2] = -2.0 / (far - near)
        proj[3,3] = 1.0

        proj[0,3] = -(right + left) / (right - left)
        proj[1,3] = -(top + bottom) / (top - bottom)
        proj[2,3] = -(far + near) / (far - near)

        return proj

    # def generate_camera_sequence_ortho(self, radius=2.0, ortho_scale=1.5, near=0.1, far=100.0):

    #     centers = np.array([0,0,0], dtype=np.float32)
    #     # up = np.array([0,1,0], dtype=np.float32)

    #     directions = [
    #         np.array([1,0,0], dtype=np.float32),   # +X
    #         np.array([-1,0,0], dtype=np.float32),  # -X
    #         np.array([0,1,0], dtype=np.float32),   # +Y
    #         np.array([0,-1,0], dtype=np.float32),  # -Y
    #         np.array([0,0,1], dtype=np.float32),   # +Z
    #         np.array([0,0,-1], dtype=np.float32),  # -Z
    #     ]
    #     up = [
    #         np.array([0,-1,0], dtype=np.float32),   # +X
    #         np.array([0,-1,0], dtype=np.float32),  # -X
    #         np.array([0,0,1], dtype=np.float32),   # +Y
    #         np.array([0,0,-1], dtype=np.float32),  # -Y
    #         np.array([0,-1,0], dtype=np.float32),   # +Z
    #         np.array([0,-1,0], dtype=np.float32),  # -Z
    #     ]
    #     self.pose = []
    #     self.proj = []

    #     left, right = -ortho_scale, ortho_scale
    #     bottom, top = -ortho_scale, ortho_scale
    #     proj = self.orthographic(left=left, right=right, bottom=bottom, top=top, near=near, far=far)

    #     for idx, d in enumerate(directions):
    #         eye = centers + d * radius
    #         pose = self.look_at(eye=eye, center=centers, up=up[idx])

    #         self.pose.append(pose)
    #         self.proj.append(proj)


    def generate_camera_sequence_ortho(self, radius=2.0, ortho_scale=1.5, near=0.1, far=100.0, azimuths=[0,0], elevations=[0,0]):

        centers = np.array([0,0,0], dtype=np.float32)
        # up = np.array([0,1,0], dtype=np.float32)

        # azimuths = [
        #     0.0,
        #     90.0,
        #     180.0,
        #     270.0,
        #     330,
        #     30,
        #     330,
        #     30,
        #     0.0,
        #     90.0,
        #     180.0,
        #     270.0,
        #     150.0,
        #     210.0,
        #     0.0,
        #     90.0,
        #     180.0,
        #     270.0,
        #     0.0,
        #     90.0,
        #     180.0,
        #     270.0,
        #     0.0,
        #     180.0
        # ]
        # elevations = [
        #     20,
        #     20,
        #     20,
        #     20,
        #     -20.0,
        #     -20.0,
        #     20,
        #     20,
        #     -20.0,
        #     -20.0,
        #     -20.0,
        #     -20.0,
        #     0.0,
        #     0.0,
        #     70.0,
        #     70.0,
        #     70.0,
        #     70.0,
        #     -70.0,
        #     -70.0,
        #     -70.0,
        #     -70.0,
        #     90.0,
        #     -90.0
        # ]
        self.pose = []
        self.proj = []
        self.w2c = []

        left, right = -ortho_scale, ortho_scale
        bottom, top = -ortho_scale, ortho_scale
        proj = self.orthographic(left=left, right=right, bottom=bottom, top=top, near=near, far=far)

        for idx, (azimuth, elevation) in enumerate(zip(azimuths, elevations)):
            azimuth = -azimuth
            # 转换为弧度
            az_rad = np.deg2rad(azimuth)
            el_rad = np.deg2rad(elevation)

            x = radius * np.cos(el_rad) * np.cos(az_rad)
            y = radius * np.sin(el_rad)
            z = radius * np.cos(el_rad) * np.sin(az_rad)

            eye = np.array([x, y, z], dtype=np.float32)

            pose = self.look_at(eye, centers)
            w2c = self.get_w2c_matrix(radius, -azimuth, elevation)
            self.w2c.append(w2c)

            self.pose.append(pose)
            self.proj.append(proj)

    def generate_random_camera_sequence(
        self,
        num_views=32,
        radius=2.0,
        ortho_scale=1.5,
        near=0.1,
        far=100.0,
        azimuth_range=(0, 360),
        elevation_range=(0, 30)
    ):
        centers = np.array([0, 0, 0], dtype=np.float32)

        self.pose = []
        self.proj = []
        self.w2c = []

        left, right = -ortho_scale, ortho_scale
        bottom, top = -ortho_scale, ortho_scale
        proj = self.orthographic(left=left, right=right, bottom=bottom, top=top, near=near, far=far)

        for _ in range(num_views):
            # 随机采样 azimuth 和 elevation，单位：度
            if np.random.rand() < 0.75:
                azimuth = np.random.uniform(-90,90)
            else:
                azimuth = np.random.uniform(90,270)
            azimuth = -azimuth
            elevation = np.random.uniform(*elevation_range)


            # 转换为弧度
            az_rad = np.deg2rad(azimuth)
            el_rad = np.deg2rad(elevation)

            # 球面坐标转笛卡尔坐标，注意这里Y轴作为up方向
            x = radius * np.cos(el_rad) * np.cos(az_rad)
            y = radius * np.sin(el_rad)
            z = radius * np.cos(el_rad) * np.sin(az_rad)

            eye = np.array([x, y, z], dtype=np.float32)

        

            pose = self.look_at(eye, centers)
            w2c = self.get_w2c_matrix(radius, -azimuth, elevation)
            self.w2c.append(w2c)

            self.pose.append(pose)
            self.proj.append(proj)


    def generate_camera_sequence_perspective(self, radius=2.0, fovy=1.0, aspect=1.0, near=0.1, far=100.0):

        centers = np.array([0,0,0], dtype=np.float32)
        # up = np.array([0,1,0], dtype=np.float32)

        azimuths = [
            0.0,
            90.0,
            180.0,
            270.0,
            0,
            0
        ]
        elevations = [
            0,
            0,
            0,
            0,
            90.0,
            -90.0
        ]
        self.pose = []
        self.proj = []
        self.w2c = []

        proj = self.perspective(fovy, aspect, near, far)

        for idx, (azimuth, elevation) in enumerate(zip(azimuths, elevations)):
            # 转换为弧度
            azimuth = -azimuth
            az_rad = np.deg2rad(azimuth)
            el_rad = np.deg2rad(elevation)

            # 球面坐标转笛卡尔坐标，注意这里Y轴作为up方向
            x = radius * np.cos(el_rad) * np.cos(az_rad)
            y = radius * np.sin(el_rad)
            z = radius * np.cos(el_rad) * np.sin(az_rad)

            eye = np.array([x, y, z], dtype=np.float32)

            pose = self.look_at(eye, centers)

            w2c = self.get_w2c_matrix(radius, -azimuth, elevation)
            self.w2c.append(w2c)

            self.pose.append(pose)
            self.proj.append(proj)


    def generate_random_camera_sequence_perspective(
        self,
        num_views=32,
        radius=2.0,
        fovy=1.0, 
        aspect=1.0,
        near=0.1,
        far=100.0,
        azimuth_range=(0, 360),
        elevation_range=(0, 30)
    ):
        centers = np.array([0, 0, 0], dtype=np.float32)

        self.pose = []
        self.proj = []
        self.w2c = []

        proj = self.perspective(fovy=fovy, aspect=aspect, near=near, far=far)

        for _ in range(num_views):
            # 随机采样 azimuth 和 elevation，单位：度
            if np.random.rand() < 0.75:
                azimuth = np.random.uniform(-90,90)
            else:
                azimuth = np.random.uniform(90,270)
            elevation = np.random.uniform(*elevation_range)

            azimuth = -azimuth
            # 转换为弧度
            az_rad = np.deg2rad(azimuth)
            el_rad = np.deg2rad(elevation)

            # 球面坐标转笛卡尔坐标，注意这里Y轴作为up方向
            x = radius * np.cos(el_rad) * np.cos(az_rad)
            y = radius * np.sin(el_rad)
            z = radius * np.cos(el_rad) * np.sin(az_rad)

            eye = np.array([x, y, z], dtype=np.float32)
            pose = self.look_at(eye, centers)

            w2c = self.get_w2c_matrix(radius, -azimuth, elevation)
            self.w2c.append(w2c)


            

            self.pose.append(pose)
            self.proj.append(proj)