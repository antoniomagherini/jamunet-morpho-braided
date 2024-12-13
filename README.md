# JAMUNET: PREDICTING THE MORPHOLOGICAL CHANGES OF BRAIDED SAND-BED RIVERS WITH DEEP LEARNING

This repository stores the data, files, code and other documents necessary for the completion of the Master's thesis of [Antonio Magherini](https://nl.linkedin.com/in/antonio-magherini-4349b2229),
student of the MSc Civil Engineering program - Hydraulic Engineering track, with a specialization in River Engineering 
at the [Faculty of Civil Engineering and Geosciences](https://www.tudelft.nl/citg) of Delft University of Technology (TU Delft).
\
The thesis was carried out in collaboration with [Deltares](https://www.deltares.nl/en). The thesis can be found at [TU Delft repository](https://repository.tudelft.nl/record/uuid:38ea0798-dd3d-4be2-b937-b80621957348).

The thesis committee of this project was composed of [Dr. Riccardo Taormina](https://www.tudelft.nl/citg/over-faculteit/afdelingen/watermanagement/medewerker/universitair-docent-onderwijzer/dr-riccardo-taormina) (Committee chairman, TU Delft), [Dr. ir. Erik Mosselman](https://www.deltares.nl/en/expertise/our-people/erik-mosselman) (Main supervisor, Deltares & TU Delft), and [Dr. ir. Víctor Chavarrías](https://www.deltares.nl/en/expertise/our-people/victor-chavarrias) (Supervisor, Deltares).

For any information, feel free to contact the author at: _antonio.magherini@gmail.com_.

The structure of this repository is the following:
- <code>benchmarks</code>, contains modules and notebooks of the benhcmark models used for comparison;
- <code>data</code>, contains raw data (satellite images, river variables);
- <code>images</code>, contains the images shown in the thesis report and other documents; (to be added soon)
- <code>model</code>, contains the modules and noteboooks with the deep-learning model;
- <code>other</code>, contains documents, images, and other files used during the project; (to be added soon)
- <code>postprocessing</code>, contains the modules used for the data postprocessing;
- <code>preliminary</code>, contains the notebooks with the preliminary data analysis, satellite image visualization, preprocessing, and other examples; (to be added soon)
- <code>preprocessing</code>, contains the modules used for the data preprocessing.

The file <code>braided.yml</code> is the environment file with all dependencies, needed to run all the notebooks.

<p align="center" style="margin-top: 1px;">
    <!-- <img src="images\jamuna_narrow_.png" alt>  -->
    <img src=".\images\1994-01-25.png" width="500"> 
</p>

<p align="center">
    <!-- <em>Jamuna River. Image taken from <a href="https://earth.google.com/web/@24.90919263,90.84277199,340.42882201a,979110.75147048d,35y,-0h,0t,0r/data=OgMKATA">Google Earth</a></em> -->
    <em>Brahmaputra-Jamuna River at the border between India and Bangladesh. The image was retrieved<br>from <a href="https://earthengine.google.com/">Google Earth Engine</a> <a href="https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C02_T1_L2">USGS Landsat 5 collection</a>. The image was taken on January 25, 1994.</em>
</p>
