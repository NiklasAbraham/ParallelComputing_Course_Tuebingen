-- Sums the squares of an array of 64-bit floats
entry sum_squares [n] (xs: [n]f64) : f64 =
  reduce (+) 0.0 (map (\x -> x*x) xs)