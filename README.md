# Unsupervised Alignment

This code is from my Machine Learning Msc thesis at UCL (Year 2019/2020). 

Abstract : This works looks at how two conceptual systems can be aligned in an unsupervised way. Concepts are here represented by two systems of D-dimensional points that show structural similarities, and we aim to find an algorithm that finds the correspondences between the two systems without the use of labels or of any known correspondences. The project is motivated by the way humans learn concepts using observed statistics from the world, and overlaps with a number of fields such as Cognitive Science, Philosophy, Neuroscience and Computer Science. A baseline algorithm inspired from the image-to-image translation literature is first tested on the datasets, showing
some but limited success.This approach is particularly sensitive to the initialization strategy, with naive initialization typically yielding solutions that become stuck in local minimums. We address this problem by adding a pre-training step that leverages information about the systems' internal structures. Local features are estimated for each point, allowing us to compute points similarities
between systems. A subset of matching points are selected based on these similarities and used to pre-train the models as if they were true correspondences. This method shows significant improvements over the baseline method. It deals notably well with concept's dimensions of up to ten, where the previous method performed poorly. It also increases success for all levels of noises explored, despite being sensitive to added noise. These results are a promising step towards showing how the brain may use the statistics from the environment to build and align conceptual systems.


#### PI: Prof. Bradley C. Love
#### Project Lead: Dr. Brett D. Roads


## Team Members

* Bradley C. Love
* Brett D. Roads
* Kaarina Aho
* Kengo Arao
* Roseline Polle


## Research Objective

Explore and develop algorithms that perform unsupervised alignment.


## Contributing

If you would like to contribute, please see See [CONTRIBUTING](CONTRIBUTING.md) for additional guidance.


## Licence

This project is licensed under the Apache Licence 2.0 - see LICENSE file for details.


## Code of Conduct

This project uses a Code of Conduct [CODE](CODE.md) adapted from the [Contributor Covenant][homepage], version 2.0, available at <https://www.contributor-covenant.org/version/2/0/code_of_conduct.html>.


# Published Work

* Roads, B.D. & Love, B.C. (2020). Learning as the unsupervised alignment of conceptual systems. *Nature Machine Intelligence*. doi: 10.1038/s42256-019-0132-2
