import os
import subprocess
import sys
import shutil

def pip_install_cookiecutter():

    """
    Installs cookiecutter library using pip package manager

    Raises:
        Exception: For cathing any errors regarding cookiecutter installation

    """
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cookiecutter==2.6.0"])
    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit()

        

def run_cookiecutter():

    """
    Runs cookiecutter command to instantiate project, using the variables within cookiecutter.json

    Raises:
        Exception: For cathing any errors while running the cookiecutter command.

    """

    try:
        command = ['python', '-m', 'cookiecutter', './', '--config-file', './cookiecutter.json', '--no-input']

        result = subprocess.run(command, check=True)
        
        if result.returncode == 0:
            print("Project was instantiated successfully.")
        else:
            print("Project was not instantiated.")
    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit()
        
def move_github_folder():

    """
    Moves .github folder into project root to be recognized by Github actions

    Raises:
        Exception: For cathing any errors regarding directory handling, for example FileNotFound.

    """
    root = os.getcwd()
    excluded_dir = '{{cookiecutter.directory_name}}'
    
    for item in os.listdir(root):
        item_path = os.path.join(root, item)
        if os.path.isdir(item_path) and item != excluded_dir:
            github_path = os.path.join(item_path, ".github")
            target_github_path = os.path.join(root, ".github")
            if os.path.isdir(github_path):
                try:
                    if os.path.exists(target_github_path):
                        shutil.rmtree(target_github_path)  # Remove existing .github folder
                    shutil.move(github_path, target_github_path)
                except Exception as e:
                    print(f"Error occurred: {e}")
                    sys.exit()
                break

def remove_template_folder():
    """
    Deletes template folder '{{cookiecutter.directory_name}}' as the use case will be developed within '{{directory_name}}' folder

    Raises:
        Exception: For cathing any errors regarding removing directory, for example FolderNotFound.

    """
    # Get the base directory of the project
    root = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the folder path to delete
    folder_to_delete = os.path.join(root, "{{cookiecutter.directory_name}}")
    
    # Check if the folder exists
    if os.path.exists(folder_to_delete):
        try:
            # Delete the folder and all its contents
            shutil.rmtree(folder_to_delete)
            print(f"Successfully deleted: {folder_to_delete}")
        except Exception as e:
            print(f"Error while deleting folder: {e}")
            sys.exit(1)
    else:
        print(f"Folder not found: {folder_to_delete}")
        sys.exit()

def main():

    """
    Main function of script

    """

    pip_install_cookiecutter() #Install cookiecutter

    run_cookiecutter() #Run cookiecutter

    move_github_folder() #Move .github folder to upper directory, into root folder

    remove_template_folder() #Deletes template folder

if __name__ == "__main__":
    main()
