pip install "zenml[server]"

zenml init
zenml up

zenml integration install mlflow -y
zenml experiment-tracker register <mlflow_tracker> --flavor=mlflow
zenml stack list
zenml stack describe

zenml model-deployer register <mlfow> --flavor=mlfow
zenml stack register <mlflow_stack> -a default -o default -d mlflow -e mlflow_tracker --set
zenml stack describe

python run_pipeline.py

# mlflow experiment tracker uri
mlflow ui --backend-store-uri "file:C:\Users\sowmy\AppData\Roaming\zenml\local_stores\5df1649e-7058-417d-aca4-b2a4c7d7f32e\mlruns"

===============================================

/mnt/e/SOWMYA/01_Ayush_ML_Course/MachineLearningEngineering/MLOps/custom-sat-mlops

pip3 install virtualenv

mkdir project
cd project

sudo virtualenv venv
source venv/bin/activate

==========================

conda config --set auto_activate_base false