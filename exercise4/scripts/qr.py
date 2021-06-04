temp_model = pickle.load(open(f"{modelsDir}/{XXXXXX[]"modelName"]}.sav", "rb"))


def listener(self):
    try:
        link = rospy.wait_for_message("/sem_nekaj", String)
        return True
    except Exception as e:
        print(e)
        return False
