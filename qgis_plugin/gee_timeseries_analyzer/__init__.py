def classFactory(iface):
    from .gee_timeseries_analyzer import GEETimeSeriesAnalyzerPlugin
    return GEETimeSeriesAnalyzerPlugin(iface)
