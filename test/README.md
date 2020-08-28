
`backend_test.cc` is modified from Ceres' example code, `simple_bundle_adjuster.cc`.

There are several main modifications to make the code extendable:
- use `ParameterBlock` to handle parameters (we can further implement local parameterization in the future)
- use `SizedCostFunction` to manage residuals, including time propagation error and observation error
- represent orientation by quaternion

### Usage
```
./backend_test [path_to_BAL_file]
```