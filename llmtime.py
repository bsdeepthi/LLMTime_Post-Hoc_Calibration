from tqdm import tqdm
from serialize import serialize_arr, deserialize_str, SerializerSettings
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from dataclasses import dataclass
from models.llms import completion_fns, nll_fns, tokenization_fns, context_lengths

STEP_MULTIPLIER = 1.2


@dataclass
class Scaler:
    """
    Represents a data scaler with transformation and inverse transformation functions.

    Attributes:
        transform (callable): Function to apply transformation.
        inv_transform (callable): Function to apply inverse transformation.
    """
    transform: callable = lambda x: x
    inv_transform: callable = lambda x: x


def get_scaler(history, alpha=0.95, beta=0.3, basic=False):
    """
    Generate a Scaler object based on given history data.

    Args:
        history (array-like): Data to derive scaling from.
        alpha (float, optional): Quantile for scaling. Defaults to .95.
        beta (float, optional): Shift parameter. Defaults to .3.
        basic (bool, optional): If True, no shift is applied, and scaling by values below 0.01 is avoided. Defaults to False.

    Returns:
        Scaler: Configured scaler object.
    """
    history = history[~np.isnan(history)]
    if basic:
        q = np.maximum(np.quantile(np.abs(history), alpha), .01)
        def transform(x):
            return x / q
        def inv_transform(x):
            return x * q
    else:
        min_ = np.min(history) - beta * (np.max(history) - np.min(history))
        q = np.quantile(history - min_, alpha)
        if q == 0:
            q = 1
        def transform(x):
            return (x - min_) / q
        def inv_transform(x):
            return x * q + min_
    return Scaler(transform=transform, inv_transform=inv_transform)


def truncate_input(input_arr, input_str, settings, model, steps):
    """
    Truncate inputs to the maximum context length for a given model.

    Args:
        input_arr (array-like): Input time series.
        input_str (str): Serialized input time series.
        settings (SerializerSettings): Serialization settings.
        model (str): Name of the LLM model to use.
        steps (int): Number of steps to predict.

    Returns:
        tuple: Truncated (input_arr, input_str).
    """
    if model in tokenization_fns and model in context_lengths:
        tokenization_fn = tokenization_fns[model]
        context_length = context_lengths[model]
        input_str_chunks = input_str.split(settings.time_sep)
        truncated_input_arr = input_arr
        truncated_input_str = input_str
        for i in range(len(input_str_chunks) - 1):
            truncated_input_str = settings.time_sep.join(input_str_chunks[i:])
            if not truncated_input_str.endswith(settings.time_sep):
                truncated_input_str += settings.time_sep
            input_tokens = tokenization_fn(truncated_input_str)
            num_input_tokens = len(input_tokens)
            avg_token_length = num_input_tokens / (len(input_str_chunks) - i)
            num_output_tokens = avg_token_length * steps * STEP_MULTIPLIER
            if num_input_tokens + num_output_tokens <= context_length:
                truncated_input_arr = input_arr[i:]
                break
        if len(truncated_input_arr) < len(input_arr):
            print(f'Warning: Truncated input from {len(input_arr)} to {len(truncated_input_arr)}')
        return truncated_input_arr, truncated_input_str
    else:
        return input_arr, input_str


def handle_prediction(pred, expected_length, strict=False):
    """
    Process the output from LLM after deserialization.

    Args:
        pred (array-like or None): The predicted values.
        expected_length (int): Expected length of the prediction.
        strict (bool, optional): If True, returns None for invalid predictions. Defaults to False.

    Returns:
        array-like: Processed prediction.
    """
    if pred is None:
        return None
    if len(pred) < expected_length:
        if strict:
            print(f'Warning: Prediction too short {len(pred)} < {expected_length}, returning None')
            return None
        else:
            print(f'Warning: Prediction too short {len(pred)} < {expected_length}, padded with last value')
            return np.concatenate([pred, np.full(expected_length - len(pred), pred[-1])])
    else:
        return pred[:expected_length]


def generate_predictions(
    completion_fn,
    input_strs,
    steps,
    settings: SerializerSettings,
    scalers,
    num_samples=1,
    temp=0.7,
    parallel=True,
    strict_handling=False,
    max_concurrent=10,
    **kwargs,
):
    """
    Generate and process text completions from a language model for input time series.

    Args:
        completion_fn (callable): Function to obtain text completions from the LLM.
        input_strs (list of str): Serialized input time series strings.
        steps (int): Number of steps to predict.
        settings (SerializerSettings): Settings for serialization.
        scalers (list of Scaler): List of Scaler objects.
        num_samples (int, optional): Number of samples to return. Defaults to 1.
        temp (float, optional): Temperature for sampling. Defaults to 0.7.
        parallel (bool, optional): If True, run completions in parallel. Defaults to True.
        strict_handling (bool, optional): Return None for bad-format predictions. Defaults to False.
        max_concurrent (int, optional): Max concurrent completions. Defaults to 10.
        **kwargs: Forwarded to completion_fn (e.g. top_p, gpt_model).

    Returns:
        tuple: (preds, completions_list, input_strs)
    """
    completions_list = []
    # FIX: forward **kwargs to completion_fn so parameters like top_p are honoured
    complete = lambda x: completion_fn(
        input_str=x,
        steps=steps * STEP_MULTIPLIER,
        settings=settings,
        num_samples=num_samples,
        temp=temp,
        **kwargs,
    )

    if parallel and len(input_strs) > 1:
        print('Running completions in parallel for each input')
        with ThreadPoolExecutor(min(max_concurrent, len(input_strs))) as p:
            completions_list = list(tqdm(p.map(complete, input_strs), total=len(input_strs)))
    else:
        completions_list = [complete(input_str) for input_str in tqdm(input_strs)]

    def completion_to_pred(completion, inv_transform):
        pred = handle_prediction(
            deserialize_str(completion, settings, ignore_last=False, steps=steps),
            expected_length=steps,
            strict=strict_handling,
        )
        if pred is not None:
            return inv_transform(pred)
        return None

    preds = [
        [completion_to_pred(completion, scaler.inv_transform) for completion in completions]
        for completions, scaler in zip(completions_list, scalers)
    ]
    return preds, completions_list, input_strs


def get_llmtime_predictions_data(
    train,
    test,
    model,
    settings,
    num_samples=10,
    temp=0.7,
    alpha=0.95,
    beta=0.3,
    basic=False,
    parallel=True,
    **kwargs,
):
    """
    Obtain forecasts from an LLM based on training series (history).

    Args:
        train (array-like or list): Training time series data (history).
        test (array-like or list): Test time series data (true future).
        model (str): LLM model key — must exist in completion_fns.
        settings (SerializerSettings or dict): Serialization settings.
        num_samples (int, optional): Number of forecast samples. Defaults to 10.
        temp (float, optional): Sampling temperature. Defaults to 0.7.
        alpha (float, optional): Scaler quantile. Defaults to 0.95.
        beta (float, optional): Scaler shift. Defaults to 0.3.
        basic (bool, optional): Use basic scaling. Defaults to False.
        parallel (bool, optional): Parallel completions. Defaults to True.
        **kwargs: Forwarded to completion_fn (e.g. top_p).

    Returns:
        dict: samples, median, info, completions_list, input_strs, optionally NLL/D.
    """
    assert model in completion_fns, f'Invalid model {model}, must be one of {list(completion_fns.keys())}'
    completion_fn = completion_fns[model]
    nll_fn = nll_fns.get(model)

    if isinstance(settings, dict):
        settings = SerializerSettings(**settings)
    if not isinstance(train, list):
        train = [train]
        test = [test]

    for i in range(len(train)):
        if not isinstance(train[i], pd.Series):
            train[i] = pd.Series(train[i], index=pd.RangeIndex(len(train[i])))
            test[i] = pd.Series(
                test[i],
                index=pd.RangeIndex(len(train[i]), len(test[i]) + len(train[i])),
            )

    test_len = len(test[0])
    assert all(len(t) == test_len for t in test), (
        f'All test series must have same length, got {[len(t) for t in test]}'
    )

    scalers = [get_scaler(train[i].values, alpha=alpha, beta=beta, basic=basic) for i in range(len(train))]

    input_arrs = [train[i].values for i in range(len(train))]
    transformed_input_arrs = np.array([
        scaler.transform(input_array) for input_array, scaler in zip(input_arrs, scalers)
    ])
    input_strs = [serialize_arr(scaled, settings) for scaled in transformed_input_arrs]
    input_arrs, input_strs = zip(*[
        truncate_input(input_array, input_str, settings, model, test_len)
        for input_array, input_str in zip(input_arrs, input_strs)
    ])

    steps = test_len
    samples = None
    medians = None
    completions_list = None

    if num_samples > 0:
        preds, completions_list, input_strs = generate_predictions(
            completion_fn, input_strs, steps, settings, scalers,
            num_samples=num_samples, temp=temp, parallel=parallel, **kwargs,
        )
        samples = [pd.DataFrame(preds[i], columns=test[i].index) for i in range(len(preds))]
        medians = [sample.median(axis=0) for sample in samples]
        samples = samples if len(samples) > 1 else samples[0]
        medians = medians if len(medians) > 1 else medians[0]

    out_dict = {
        'samples': samples,
        'median': medians,
        'info': {'Method': model},
        'completions_list': completions_list,
        'input_strs': input_strs,
    }

    if nll_fn is not None:
        BPDs = [
            nll_fn(
                input_arr=input_arrs[i],
                target_arr=test[i].values,
                settings=settings,
                transform=scalers[i].transform,
                count_seps=True,
                temp=temp,
            )
            for i in range(len(train))
        ]
        out_dict['NLL/D'] = np.mean(BPDs)

    return out_dict
