import datasets

if __name__ == '__main__':
    glue_dataset = datasets.load_dataset(path="glue", name="mrpc")
    glue_dataset.save_to_disk("data")
