import wandb


wandb.login(key="690a936311e63ff7c923d0a2992105f537cd7c59")
run = wandb.init(name = "xvector model", project="huggingface")

artifact = wandb.Artifact('model', type='model')
artifact.add_file(local_path="/nlsasfs/home/nltm-st/sujitk/yash/Wav2vec-codes/xVectorResults/modelEpoch0_xVector.pth")
run.log_artifact(artifact)
run.finish()
print("Finished uploading the artifact.")