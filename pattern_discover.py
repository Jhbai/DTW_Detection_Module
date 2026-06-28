def normal_pattern_discover(data, similarity, minlen, maxlen, k):
    from dtaidistance.subsequence.dtw import local_concurrences

    lc = local_concurrences(data, None, estimate_settings=similarity, use_c=True)
    lc.align()
    matches_obj = lc.kbest_matches_store(k=k, minlen=minlen)

    templates = list()
    for st, ed in matches_obj.segments()[0]:
      if ed - st < maxlen:
        templates += [data[st:ed]]
    return templates
