import captum
import matplotlib.pyplot as plt
import numpy as np
import stable_baselines3
import torch as th
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from data_collection import framework_selector
from data_collection.sb_zoo_connector import StableBaselines3Agent
from matplotlib.backends.backend_agg import FigureCanvas
from PIL import Image
from skimage import exposure
from skimage.transform import resize
from stable_baselines3.common.policies import ActorCriticPolicy


def plt_fig2data(figure: plt.Figure):
    figure.tight_layout(pad=0)
    figure.canvas.draw()

    data = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))

    return data


def create_attribution_overlay_image(original, attributions):
    fig, _ = viz.visualize_image_attr(
        attributions,
        exposure.adjust_gamma(original, gamma=0.5, gain=1),
        method="blended_heat_map",
        show_colorbar=True,
        sign="all",
        use_pyplot=False,
        cmap="seismic",
        alpha_overlay=0.7,
        fig_size=(4, 4),
    )
    fig.tight_layout(pad=0)
    canvas = FigureCanvas(fig)
    canvas.draw()
    return Image.fromarray(np.array(canvas.renderer.buffer_rgba()))


def get_input_attribution_for_image(obs, action, exp_id, checkpoint_step, framework):
    """
    Get the input attribution for an image.
    :param obs: The model input.
    :param exp_id: The experiment id.
    :param checkpoint_step: The checkpoint step.
    :return:
    """

    assert framework == "StableBaselines3", "Framework not supported"

    agent: StableBaselines3Agent = framework_selector.get_agent(framework=framework)(
        None, None, exp=exp_id, device="auto", checkpoint_step=checkpoint_step
    )

    def logits_from_model(obs_inp):
        if hasattr(agent.model, "policy"):
            return agent.model.policy.action_net(
                agent.model.policy.mlp_extractor(
                    agent.model.policy.extract_features(obs_inp)
                )[0]
            )
        else:
            return agent.model.action_net(
                agent.model.mlp_extractor(agent.model.extract_features(obs_inp))[0]
            )

    explainer = IntegratedGradients(logits_from_model)

    target = action if np.isscalar(action) else action.argmax()

    attr_buffer = (
        explainer.attribute(
            agent.model.policy.obs_to_tensor(np.expand_dims(obs, 0))[0], target=target
        )
        .detach()
        .cpu()
        .numpy()
    )

    # reorder channels to match image
    attr_buffer = attr_buffer.squeeze().transpose(1, 2, 0)

    return create_attribution_overlay_image(obs, attr_buffer)
