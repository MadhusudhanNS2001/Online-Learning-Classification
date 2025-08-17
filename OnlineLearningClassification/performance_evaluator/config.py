import os
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

root_path = os.path.dirname(__file__)

# Default
FONT = 'JetBrainsMono'
FONT_STYLE = {
    'Roboto': ['Light', 'Regular', 'Medium', 'Bold'],
    'JetBrainsMono': ['Light', 'Regular', 'SemiBold', 'Bold'],
    'FiraCode': ['Light', 'Regular', 'Medium', 'Bold'],
}
COLOR_MAPS = plt.colormaps()
CURRENT_CMAP = 'tab20'

# Confusion Matrix
CONFUSION_MATRIX = dict(
    cmap=CURRENT_CMAP,
    cbar=False,
    cbar_ticklabels_fontdict=dict(
        font=Path(root_path, 'fonts/{0}/{0}-{1}.ttf'.format(FONT, FONT_STYLE[FONT][1])),
        size=13
    ),
    annot_kws=dict(
        font=Path(root_path, 'fonts/{0}/{0}-{1}.ttf'.format(FONT, FONT_STYLE[FONT][1])),
        size=13
    ),
    xticklabels_rotation=0,
    yticklabels_rotation=90,
    ticklabels_fontdict=dict(
        font=Path(root_path, 'fonts/{0}/{0}-{1}.ttf'.format(FONT, FONT_STYLE[FONT][1])),
        size=13
    ),
    title='Confusion Matrix',
    titlepad=10,
    title_fontdict=dict(
        font=Path(root_path, 'fonts/{0}/{0}-{1}.ttf'.format(FONT, FONT_STYLE[FONT][1])),
        size=13
    ),
    xlabel='Predicted Class',
    ylabel='Actual Class',
    xylabelpad=10,
    xylabel_fontdict=dict(
        font=Path(root_path, 'fonts/{0}/{0}-{1}.ttf'.format(FONT, FONT_STYLE[FONT][1])),
        size=13
    ),
)

# Precision Recall Curve
PR_CURVE = dict(
    ticklabels_fontdict=dict(
        font=Path(root_path, 'fonts/{0}/{0}-{1}.ttf'.format(FONT, FONT_STYLE[FONT][1])),
        size=13
    ),
    title='Precision-Recall Curve',
    titlepad=10,
    title_fontdict=dict(
        font=Path(root_path, 'fonts/{0}/{0}-{1}.ttf'.format(FONT, FONT_STYLE[FONT][1])),
        size=13
    ),
    xlabel='Recall',
    ylabel='Precision',
    xylabelpad=10,
    xylabel_fontdict=dict(
        font=Path(root_path, 'fonts/{0}/{0}-{1}.ttf'.format(FONT, FONT_STYLE[FONT][1])),
        size=13
    ),
    legend_fontdict=dict(
        family=FontProperties(
            fname=os.path.join(root_path, 'fonts/{0}/{0}-{1}.ttf'.format(FONT, FONT_STYLE[FONT][1]))
        ).get_name(),
        size=13
    ),
    legend_ncol=1,
)

# Receiver Operating Characteristic Curve
ROC_CURVE = dict(
    ticklabels_fontdict=dict(
        font=Path(root_path, 'fonts/{0}/{0}-{1}.ttf'.format(FONT, FONT_STYLE[FONT][1])),
        size=12
    ),
    title='Receiver Operating Characteristic Curve',
    titlepad=15,
    title_fontdict=dict(
        font=Path(root_path, 'fonts/{0}/{0}-{1}.ttf'.format(FONT, FONT_STYLE[FONT][1])),
        size=16
    ),
    xlabel='False Positive Rate',
    ylabel='True Positive Rate',
    xylabelpad=15,
    xylabel_fontdict=dict(
        font=Path(root_path, 'fonts/{0}/{0}-{1}.ttf'.format(FONT, FONT_STYLE[FONT][1])),
        size=14
    ),
    legend_fontdict=dict(
        family=FontProperties(
            fname=os.path.join(root_path, 'fonts/{0}/{0}-{1}.ttf'.format(FONT, FONT_STYLE[FONT][1]))
        ).get_name(),
        size=12
    ),
    legend_ncol=1,
)
