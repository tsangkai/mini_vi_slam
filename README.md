# Mini Visual-Inertial SLAM


### Build instruction

To build this project, just follow standard CMake procedure.
```
mkdir build
cd build
cmake .. [-D BUILD_TEST=OFF]
cmake --build .
```

### Usage

Please check the configuration file `config/test.yaml` for the workflow including both frontend and backend.

To run `frontend`, please provide the directory of images.
```
./frontend [path_to_image_directory]
```

To run `backend`, one needs to sepcify three files: observation file, IMU file, and ground truth for initialization.
```
./backend [path_to_observation_file] [path_to_imu_file] [path_to_ground_truthw_file]
```

Currently, for developing, I write the paths in the code. Therefore, directly executing those files without argements is fine.