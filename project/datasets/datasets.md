# Datasets

## Galaxy10 SDSS

[LINK](https://zenodo.org/records/10844811)

Galaxy10 SDSS is a dataset contains 21785 69x69 pixels colored galaxy images (g, r and i band) separated in 10 classes. Galaxy10 SDSS images come from Sloan Digital Sky Survey and labels come from Galaxy Zoo.

These classes are mutually exclusive, but Galaxy Zoo relies on human volunteers to classify galaxy images and the volunteers do not agree on all images. For this reason, Galaxy10 only contains images for which more than 55% of the votes agree on the class. That is, more than 55% of the votes among 10 classes are for a single class for that particular image. If none of the classes get more than 55%, the image will not be included in Galaxy10 as no agreement was reached. As a result, 21785 images after the cut.

The justification of 55% as the threshold is based on validation. Galaxy10 is meant to be an alternative to MNIST or Cifar10 as a deep learning toy dataset for astronomers. Thus astroNN.models.Cifar10_CNN is used with Cifar10 as a reference. The validation was done on the same astroNN.models.Cifar10_CNN. 50% threshold will result a poor neural network classification accuracy although around 36000 images in the dataset, many are probably misclassified and neural network has a difficult time to learn. 60% threshold result is similar to 55% , both classification accuracy is similar to Cifar10 dataset on the same network, but 55% threshold will have more images be included in the dataset. Thus 55% was chosen as the threshold to cut data.

The original images are 424x424, but were cropped to 207x207 centered at the images and then downscaled 3 times via bilinear interpolation to 69x69 in order to make them manageable on most computer and graphics card memory.

There is no guarantee on the accuracy of the labels. Moreover, Galaxy10 is not a balanced dataset and it should only be used for educational or experimental purpose. If you use Galaxy10 for research purpose, please cite Galaxy Zoo and Sloan Digital Sky Survey.

For more information on the original classification tree: Galaxy Zoo Decision Tree.

---

## Galaxy10 DECaLS

[LINK](https://zenodo.org/records/10845026)

The original Galaxy10 dataset was created with Galaxy Zoo (GZ) Data Release 2 where volunteers classify ~270k of SDSS galaxy images where ~22k of those images were selected in 10 broad classes using volunteer votes. GZ later utilized images from DESI Legacy Imaging Surveys (DECaLS) with much better resolution and image quality. Galaxy10 DECaLS has combined all three (GZ DR2 with DECals images instead of SDSS images and DECaLS campaign ab, c) results in ~441k of unique galaxies covered by DECaLS where ~18k of those images were selected in 10 broad classes using volunteer votes with more rigorous filtering. Galaxy10 DECals had its 10 broad classes tweaked a bit so that each class is more distinct from each other and Edge-on Disk with Boxy Bulge class with only 17 images in original Galaxy10 was abandoned. The source code for this dataset is released under this repository so you are welcome to play around if you like, otherwise you can use the compiled Galaxy10 DECaLS with download link below.

### Galaxy Zoo

[LINK](https://www.zooniverse.org/projects/zookeeper/galaxy-zoo)

---

|                  | SDSS      | DECaLS                     |
| ---------------- | --------- | -------------------------- |
| Imagens          | 21.785    | 17.675 (pós-deduplicação)  |
| Resolução        | 69×69     | 256×256                    |
| Fonte            | SDSS      | DESI Legacy Surveys        |
| Bandas           | g, r, i   | g, r, z                    |
| Tamanho          | 201 MB    | 2.6 GB                     |
| Metadados extras | —         | ra, dec, redshift, pxscale |
| Filtragem        | 55% votes | Mais rigorosa              |

---

## Estrutura dos Arquivos H5

### Galaxy10 SDSS (`Galaxy10.h5`)

| Campo    | Shape         | Dtype  | Descrição                          |
| -------- | ------------- | ------ | ---------------------------------- |
| `images` | (21785,69,69,3) | uint8  | Imagens em bandas g, r, i (0–255) |
| `ans`    | (21785,)      | uint8  | Labels de classe (0–9)             |

### Galaxy10 DECaLS (`Galaxy10_DECals_NoDuplicated.h5`)

| Campo       | Shape         | Dtype   | Range                | Descrição                              |
| ----------- | ------------- | ------- | -------------------- | -------------------------------------- |
| `images`    | (17675,256,256,3) | uint8 | 0–255              | Imagens em bandas g, r, z              |
| `ans`       | (17675,)      | uint8   | 0–9                  | Labels de classe                       |
| `redshift`  | (17675,)      | float64 | −0.000124 – 1.4416   | Redshift espectroscópico; 92 NaN (~0.5%) |
| `ra`        | (17675,)      | float64 | 0.01° – 359.95°      | Ascensão reta (graus)                  |
| `dec`       | (17675,)      | float64 | −19.05° – 69.77°     | Declinação (graus)                     |
| `pxscale`   | (17675,)      | float64 | 0.262 – 0.524 ″/px   | Escala de pixel (arcsec/pixel)         |

---

## Distribuição de Classes

### SDSS — desbalanceado

| Classe | Nome                   | Count  | %     | Nota                        |
| ------ | ---------------------- | ------ | ----- | --------------------------- |
| 0      | Disturbed              | 3.461  | 15.9% |                             |
| 1      | Merging                | 6.997  | 32.1% | Classe majoritária          |
| 2      | Round Smooth           | 6.292  | 28.9% | Classe majoritária          |
| 3      | In-between Smooth      | 349    | 1.6%  |                             |
| 4      | Cigar Shaped           | 1.534  | 7.0%  |                             |
| 5      | Edge-on Boxy Bulge     | 17     | 0.1%  | Abandonada no DECaLS        |
| 6      | Barred Spiral          | 589    | 2.7%  |                             |
| 7      | Unbarred Tight Spiral  | 1.121  | 5.1%  |                             |
| 8      | Unbarred Loose Spiral  | 906    | 4.2%  |                             |
| 9      | Edge-on w/ Bulge       | 519    | 2.4%  |                             |

### DECaLS — relativamente balanceado

| Classe | Nome                   | Count  | %     | Nota                  |
| ------ | ---------------------- | ------ | ----- | --------------------- |
| 0      | Disturbed              | 1.081  | 6.1%  |                       |
| 1      | Merging                | 1.853  | 10.5% |                       |
| 2      | Round Smooth           | 2.640  | 14.9% |                       |
| 3      | In-between Smooth      | 2.021  | 11.4% |                       |
| 4      | Cigar Shaped           | 333    | 1.9%  | Classe minoritária    |
| 5      | Barred Spiral          | 2.040  | 11.5% |                       |
| 6      | Unbarred Tight Spiral  | 1.823  | 10.3% |                       |
| 7      | Unbarred Loose Spiral  | 2.600  | 14.7% |                       |
| 8      | Edge-on w/o Bulge      | 1.419  | 8.0%  |                       |
| 9      | Edge-on w/ Bulge       | 1.865  | 10.6% |                       |
