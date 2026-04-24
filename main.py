import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class TaskApp:
    def __init__(self):
        self.file_name = "tasks.csv"
        self.vectorizer = TfidfVectorizer()
        self.clf = MultinomialNB()
        
        # Expanded real-world training data for better accuracy
        self.train_data = {
            "text": [
                # High Priority
                "fix production bug", "submit internship report", "emergency meeting", 
                "client deadline today", "server crash recovery", "pay rent now",
                # Medium Priority
                "read documentation", "weekly team sync", "prepare presentation", 
                "update portfolio website", "respond to emails", "grocery shopping",
                # Low Priority
                "watch Netflix", "scroll social media", "buy new shoes", 
                "casual coffee with friend", "organize desk", "check hobby blog"
            ],
            "label": [
                "High", "High", "High", "High", "High", "High",
                "Medium", "Medium", "Medium", "Medium", "Medium", "Medium",
                "Low", "Low", "Low", "Low", "Low", "Low"
            ],
        }
        self.load_tasks()

    def load_tasks(self):
        if os.path.exists(self.file_name):
            self.tasks = pd.read_csv(self.file_name)
        else:
            self.tasks = pd.DataFrame(columns=["task", "priority"])

    def save_tasks(self):
        self.tasks.to_csv(self.file_name, index=False)

    def train_model(self):
        # Trains the model using the expanded dataset
        x_train = self.vectorizer.fit_transform(self.train_data["text"])
        self.clf.fit(x_train, self.train_data["label"])

    def add_task(self, task_desc):
        self.train_model()
        # Transform the user input to match the training format
        x_input = self.vectorizer.transform([task_desc.lower()])
        predicted_priority = self.clf.predict(x_input)[0]

        new_row = pd.DataFrame({"task": [task_desc], "priority": [predicted_priority]})
        self.tasks = pd.concat([self.tasks, new_row], ignore_index=True)
        self.save_tasks()
        print(f"\n[✔] Task added!")
        print(f"[AI] ML suggestion: {predicted_priority} priority.")

    def list_tasks(self):
        if self.tasks.empty:
            print("\n[!] Your list is empty.")
        else:
            print("\n--- CURRENT TASK LIST ---")
            # Using to_string() makes the output look cleaner in the console
            print(self.tasks.to_string(index=True))

    def remove_task(self, index):
        if self.tasks.empty:
            print("\n[!] Your list is empty.")
            return

        if index < 0 or index >= len(self.tasks):
            print("\n[X] Invalid index.")
            return

        self.tasks = self.tasks.drop(index).reset_index(drop=True)
        self.save_tasks()
        print("\n[✔] Task removed successfully.")

def menu():
    app = TaskApp()
    while True:
        print("\n========================")
        print(" TASK MANAGEMENT SYSTEM ")
        print("========================")
        print("1. Add Task")
        print("2. Remove Task")
        print("3. List All Tasks")
        print("4. Exit")
        
        choice = input("\nSelect an option: ")

        if choice == "1":
            desc = input("Enter task description: ")
            app.add_task(desc)
        elif choice == "2":
            app.list_tasks()
            try:
                idx = int(input("\nEnter index number to remove: "))
                app.remove_task(idx)
            except ValueError:
                print("[X] Error: Please enter a valid numerical ID.")
        elif choice == "3":
            app.list_tasks()
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("[X] Invalid choice. Try again.")

if __name__ == "__main__":
    menu()