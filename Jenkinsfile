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

        stage('Test') {
            steps {
                sh 'mkdir result'
                sh 'pytest test/test_binding.py --junitxml=test_result/test_binding.py.xml'
                sh 'pytest test/test_object.py --junitxml=test_result/test_object.py.xml'
                sh 'pytest test/test_render.py --junitxml=test_result/test_render.py.xml'
                sh 'pytest test/test_simulator.py --junitxml=test_result/test_simulator.py.xml'
                sh 'pytest test/test_navigate_env.py --junitxml=test_result/test_navigate_env.py.xml'
            }
        }
    
    }
    post { 
        always { 
            junit 'test_result/*.xml'
            cleanWs()
        }
    }
}