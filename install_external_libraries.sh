# From https://davidlapous.github.io/multipers/compilation.html

# more dependencies are needed
conda install llvm-openmp cgal cgal-cpp gmp mpfr eigen cmake -c conda-forge

# Temp dir
mkdir temp
cd temp

# mpfree
git clone https://bitbucket.org/mkerber/mpfree/
cd mpfree
cmake .
make
cp mpfree $CONDA_PREFIX/bin/
cd ..

# function_delaunay
git clone https://bitbucket.org/mkerber/function_delaunay/
cd function_delaunay
sed -i "8i find_package(Eigen3 3.3 REQUIRED NO_MODULE)\nlink_libraries(Eigen3::Eigen)\n" CMakeLists.txt
cmake .
make
cp main $CONDA_PREFIX/bin/function_delaunay
cd ..

# Clean up
cd ..
rm -rf temp
