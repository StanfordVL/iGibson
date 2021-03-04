pipeline {

    agent {
        docker {
            image 'gibsonchallenge/gibsonv2:jenkins'
            args '--runtime=nvidia -u root:root -v ${WORKSPACE}/../ig_dataset:${WORKSPACE}/gibson2/data/ig_dataset'
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

        stage('Build Docs') {
            steps {
                sh 'sphinx-apidoc -o docs/apidoc gibson2 gibson2/external gibson2/utils/data_utils/'
                sh 'sphinx-build -b html docs _sites'
            }
        }

        stage('Test') {
            steps {
                sh 'mkdir result'
                sh 'pytest gibson2/test/test_binding.py --junitxml=test_result/test_binding.py.xml'
                sh 'pytest gibson2/test/test_render.py --junitxml=test_result/test_render.py.xml'
                sh 'pytest gibson2/test/test_pbr.py --junitxml=test_result/test_pbr.py.xml'
                sh 'pytest gibson2/test/test_object.py --junitxml=test_result/test_object.py.xml'
                sh 'pytest gibson2/test/test_simulator.py --junitxml=test_result/test_simulator.py.xml'
                sh 'pytest gibson2/test/test_igibson_env.py --junitxml=test_result/test_igibson_env.py.xml'
                sh 'pytest gibson2/test/test_scene_importing.py --junitxml=test_result/test_scene_importing.py.xml'
                sh 'pytest gibson2/test/test_robot.py --junitxml=test_result/test_robot.py.xml'
                sh 'pytest gibson2/test/test_igsdf_scene_importing.py --junitxml=test_result/test_igsdf_scene_importing.py.xml'
                sh 'pytest gibson2/test/test_sensors.py --junitxml=test_result/test_sensors.py.xml'
                sh 'pytest gibson2/test/test_motion_planning.py --junitxml=test_result/test_motion_planning.py.xml'
                sh 'pytest gibson2/test/test_states.py --junitxml=test_result/test_states.py.xml'
            }
        }

        stage('Benchmark') {
            steps {
                sh 'python gibson2/test/benchmark/benchmark_static_scene.py'
                sh 'python gibson2/test/benchmark/benchmark_interactive_scene.py'
            }
        }
    
    }
    post { 
        always { 
            junit 'test_result/*.xml'
            archiveArtifacts artifacts: 'test_result/*.xml', fingerprint: true
            archiveArtifacts artifacts: '*.pdf'
            archiveArtifacts artifacts: '*.png'

            publishHTML (target: [
              allowMissing: true,
              alwaysLinkToLastBuild: false,
              keepAll: true,
              reportDir: '_sites',
              reportFiles: 'index.html',
              includes: '**/*',
              reportName: "iGibson docs"
            ])

            cleanWs()
        }
        failure {
            script {
                    // Send an email only if the build status has changed from green/unstable to red
                    emailext subject: '$DEFAULT_SUBJECT',
                        body: '$DEFAULT_CONTENT',
                        recipientProviders: [
                            [$class: 'CulpritsRecipientProvider'],
                            [$class: 'DevelopersRecipientProvider'],
                            [$class: 'RequesterRecipientProvider']
                        ], 
                        replyTo: '$DEFAULT_REPLYTO',
                        to: '$DEFAULT_RECIPIENTS'
                }
            }
        }
}