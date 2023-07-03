"""googlemaps_images dataset."""

import tensorflow_datasets as tfds
import pandas as pd
import os

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for googlemaps_images dataset."""

  VERSION = tfds.core.Version('1.0.4')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
      '1.0.1': 'Added all classes.',
      '1.0.2': 'Shuffled data.',
      '1.0.3': 'Shuffled data.',
      '1.0.4': 'Get satelite map.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(googlemaps_images): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(256, 256, 3)),
            #'label': tfds.features.ClassLabel(names=['no', 'yes']),
            'label': tfds.features.ClassLabel(names=['ocean','nature','farm','city','roads']),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://github.com/crazynds/FindHumanTouchIA',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = dl_manager.extract('./extra.zip')

    cities = pd.read_csv(path / "./extra/cities.csv")
    del cities['Unnamed: 0']
    random = pd.read_csv(path / "./extra/random.csv")
    del random['Unnamed: 0']


    all = pd.concat([random,cities],ignore_index=True)
    all = all.sample(frac=1)


    return {
        'train': self._generate_examples(all,path,'simple'),
        #'train': self._generate_examples(all,path,'satelite'),
    }

  def _generate_examples(self, df,path, type):
    noArr = ['nature', 'ocean']
    for tuple in df.itertuples():
      if tuple.type != type:
        continue
      yield tuple.Index, {
          'image': path / tuple.local,
          #'label': 'yes' if tuple.result not in noArr  else 'no',
          'label': tuple.result
      }
