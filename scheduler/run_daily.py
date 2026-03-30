# import schedule
# import time
# import os

# def job():
#     print("Running batch prediction...")
#     os.system('python "E:/College/infosys springboard project/Code/batch_predict.py"')

# # TEST MODE
# schedule.every(10).seconds.do(job)

# while True:
#     schedule.run_pending()
#     time.sleep(1)

import schedule
import time
import os

def job():
    print("Running batch prediction...")

    script_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "batch", "batch_predict.py")
    )

    os.system(f'python "{script_path}"')

# TEST MODEs
schedule.every(1800).seconds.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)