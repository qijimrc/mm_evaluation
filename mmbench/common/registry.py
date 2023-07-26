
class Registry:
    mapping = {
        "task_name_mapping": {},
        "metric_name_mapping": {}
    }

    @classmethod
    def register_task(cls, name):
        r"""Register a task to registry with key 'name'

        Args:
            @name: Key with which the task will be registered.
            @level: the task level.

        Usage:

            from mmbench.common.registry import registry
        """

        def wrap(task_cls):
            from mmbench.tasks.base_task import BaseTask

            assert issubclass(
                task_cls, BaseTask
            ), "All tasks must inherit BaseTask class"
            if name in cls.mapping["task_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["task_name_mapping"][name]
                    )
                )
            cls.mapping["task_name_mapping"][name] = task_cls
            return task_cls

        return wrap


    @classmethod
    def register_metric(cls, name):
        r"""Register a metric to registry with key 'name'

        Args:
            name: Key with which the metric will be registered.

        Usage:

            from mmbench.common.registry import registry
        """

        def wrap(metric_cls):
            from mmbench.metrics.base_metric import BaseMetric

            assert issubclass(
                metric_cls, BaseMetric
            ), "All metrics must inherit BaseMetric class"
            if name in cls.mapping["metric_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["metric_name_mapping"][name]
                    )
                )
            cls.mapping["metric_name_mapping"][name] = metric_cls
            return metric_cls

        return wrap
    

    @classmethod
    def get_task_class(cls, name):
        return cls.mapping["task_name_mapping"].get(name, None)

    @classmethod
    def list_tasks(cls):
        return sorted(cls.mapping["task_name_mapping"].keys())
    
    @classmethod
    def get_metric_class(cls, name):
        return cls.mapping["metric_name_mapping"].get(name, None)

    @classmethod
    def list_metrics(cls):
        return sorted(cls.mapping["metric_name_mapping"].keys())


registry = Registry()
