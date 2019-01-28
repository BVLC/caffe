pipeline {
  agent any
  stages {
    stage('Build GPU') {
      steps {
        sh '''ln -s Makefile.config.gpu.cudnn Makefile.config
make -j12'''
      }
    }
    stage('Build CPU') {
      steps {
        sh '''make clean
rm Makefile.config
ln -s Makefile.config.cpu Makefile.config
make -j24'''
      }
    }
  }
}