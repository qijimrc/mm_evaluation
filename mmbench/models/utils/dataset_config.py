from .misc import listinstr


def DATASET_TYPE(dataset):
    if listinstr(['mmbench', 'seedbench', 'ccbench', 'mmmu', 'scienceqa', 'ai2d'], dataset.lower()):
        return 'multi-choice'
    elif 'MME' in dataset:
        return 'Y/N'
    elif 'COCO' in dataset:
        return 'Caption'
    elif listinstr(['ocrvqa', 'textvqa', 'chartqa', 'mathvista', 'docvqa'], dataset.lower()):
        return 'VQA'
    else:
        return 'QA'
