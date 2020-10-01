pipeline {
    agent {
        docker {
            image 'gibsonchallenge/gibsonv2:jenkins'
            args '--runtime=nvidia -u root:root'
        }
    }
    stages {
        stage('Build') {
            steps {
                sh 'nvidia-smi'
                sh 'pwd'
                sh 'printenv'
                sh 'pip install -e .'
            }
        }
    }
    post { 
        always { 
            cleanWs()
        }
    }
}