-- Sums the squares of an array of 64-bit floats
entry sum_squares [n] (xs: [n]f64) : f64 =
  reduce (+) 0.0 (map (\x -> x*x) xs)

let matmul [n][k][m] (a: [n][k]f64) (b: [k][m]f64) : [n][m]f64 =
  map (\row ->
        map (\col ->
              reduce (+) 0.0 (map2 (*) row col))
            (transpose b))
      a

entry matmul_bias_relu_sum [n][k][m]
  (a: [n][k]f64)
  (b: [k][m]f64)
  (bias: [m]f64) : f64 =
  let c = matmul a b
  let c_bias = map (\row -> map2 (+) row bias) c
  let relu_c = map (map (\x -> if x > 0.0 then x else 0.0)) c_bias
  in reduce (+) 0.0 (map (reduce (+) 0.0) relu_c)