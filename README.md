# BoxEmbeddingCBM
Box embedding integrated in CBM architecture to take advantage of concept hierarchy representation

# Instruction to reproduce the experiments

## Download awa2 dataset

you can download the dataset directly from the website https://cvml.ista.ac.at/AwA2/

or you can lunch this command

```
python /scripts/download_awa2.py --target [choice]
```

where choice can be 'dataset', 'features', 'labels', 'all'

## Manipulation of awa2 dataset

To reproduce the experiments as described in Chapther 5 of the Thesis you should launch the following command:

```
python scripts/awa2_kg_projection.py --concepts [path] --matrix [path] --labels [path]
```

where you should substitute the path respectively with the paths of predicates.txt, predicate-matrix-binary.txt and AwA2-labels.txt files downloaded in the previous step

## 3. Run the experiments in the notebooks

Now you should be able to run the notebooks in /notebooks/awa2 to reproduce the experiments and see the results described in the Thesis pdf