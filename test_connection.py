import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

# Initialize Roboflow using environment / stored credentials
rf = Roboflow()

workspace = rf.workspace()
print(f"Connected to Workspace: {workspace.name} ({workspace.url})")

project = workspace.project("m1_m2")
print(f"Connected to Project: {project.name} (type: {project.type})")
print("Available versions:", [v.version for v in project.versions()])

