docker build -t gibsonchallenge/gibsonv2:jenkins2 --build-arg USER_ID=$(id -u jenkins) --build-arg GROUP_ID=$(id -g jenkins) .
