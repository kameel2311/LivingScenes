def collection_generator(dataloader, config):
    collection_count = 0
    number_collections = config["object_collection"]["number_collections"]
    excluded_classes = config["object_collection"]["excluded_classes"]

    # Setting the Data Loader
    object_collection_settings = config["object_collection"][
        "object_collection_settings"
    ]
    object_collection_sampler = config["object_collection"]["object_collection_sampler"]

    # Filter out the excluded classes
    allowed_classes = [
        semantic_class
        for semantic_class in dataloader.object_classes
        if semantic_class not in excluded_classes
    ]

    # Generate Object Collections
    while collection_count < number_collections:
        if object_collection_sampler["allow_same_class"]:
            semantic_classes = random.choices(
                allowed_classes,
                k=object_collection_sampler["number_objects_per_collection"],
            )
        else:
            assert object_collection_sampler["number_objects_per_collection"] <= len(
                allowed_classes
            ), "Number of objects per collection is greater than the number of allowed classes"
            semantic_classes = random.sample(
                allowed_classes,
                k=object_collection_sampler["number_objects_per_collection"],
            )

        semantic_idxs = random.choices(
            list(range(dataloader.object_idx_limit)),
            k=object_collection_sampler["number_objects_per_collection"],
        )
        objects = []
        for semantic_class, object_idx in zip(semantic_classes, semantic_idxs):
            objects.append(
                parse_scene_object(
                    config, dataloader, camera, semantic_class, object_idx
                )
            )
        object_collection = ObjectCollection(objects, object_collection_settings)
        yield object_collection

        # Increment the Collection Count
        collection_count += 1
