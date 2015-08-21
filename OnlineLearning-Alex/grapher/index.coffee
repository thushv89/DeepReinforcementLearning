fs = require 'fs'
_  = require 'lodash'
d3 = require 'd3'

data_dirs = do () ->
    basepath = '../output'
    root = fs.readdirSync basepath

    # filter out the pretraining folders
    for x in root when fs.statSync(path = basepath + '/' + x).isDirectory() and (x.indexOf 'pretrain') is -1 and (fs.existsSync(path + '/validation.csv'))
        # set a filter
        #if not /Selective/.test(x) and not /combined/.test(x)
            #continue

        if /NoPolicy/.test(x)
            continue

        if /mps/.test(x)
            continue

        if /HardPooler/.test(x)
            continue

        if not /mnist/.test(x)
            continue

        console.log(x)

        [ data_set, arch, effect, seed, model ] = x.split '_'

        'name': x,
        'path': path,
        'data_set': data_set,
        'arch': arch,
        'effect': effect,
        'seed': seed,
        'model': model
        'marginalised': not (model.indexOf('-ma') is -1)

# make the directories if they don't exist
create_dir = (dir) ->
    try
        fs.mkdirSync dir
    catch
        null

# build the chart
build_chart = (data, charter, params) ->
    width = 800
    height = 600

    svg = d3.select('body')
        .append('svg')
        .attr('xmlns', 'http://www.w3.org/2000/svg')
        .attr('width', 800)
        .attr('height', 600)

    # run the charter
    charter data, width, height, svg, params

    data = d3.select('body').html()
    d3.select('svg').remove()
    return data

create_margin = (margin, width, height, svg) ->
    width = width - margin.left - margin.right
    height = height - margin.top - margin.bottom

    # create the chart
    svg.append('g')
        .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')')

    chart = svg.selectAll('g')
    return [width, height, chart]

create_axes = (x, y, width, height, margin, x_axis, y_axis, x_title, y_title, svg, chart) ->
    # axes
    chart.append('g')
        .attr('class', 'x axis')
        .attr('transform', 'translate(0, ' + height + ')')
        .call(x_axis)

    chart.append('g')
        .attr('class', 'y axis')
        .call(y_axis)

    svg.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', - height / 2 - margin.top)
        .attr('y', '1em')
        .style('dy', '1em')
        .style('text-anchor', 'middle')
        .attr('class', 'axis-label')
        .text(y_title)

    svg.append('text')
        .attr('x', width / 2 + margin.left)
        .attr('y', height + margin.bottom)
        .style('text-anchor', 'middle')
        .attr('class', 'axis-label')
        .text(x_title)

# stacked bar chart
stacked_bar = (data, width, height, svg, params) ->
    # margins
    margin =
        'top': 20
        'bottom': 70
        'right': 20
        'left': 70

    [ width, height, chart ] = create_margin margin, width, height, svg

    # svg style sheet
    svg_style = svg.append('defs')
        .append('style')
        .attr('type','text/css')

    css_text = '<![CDATA[
                    text { font: 12px sans-serif; }
                    .axis path, .axis line { fill: none; stroke: black; shape-rendering: crispEdges; }
                ]]>'

    svg_style.text css_text

    ################################
    # data preparation

    # find all the labels, and patch up the data with zeros when neccessary
    labels = { }
    for item in data
        for key in Object.keys(item)
            labels[key] = 1

    for item in data
        for key of labels when key not of item
            item[key] = 0.0

    # transpose the data set
    new_set = ({ 'name': label, 'values': [] } for label of labels)

    data.forEach (item, i) ->
        for key, value of item
            new_set[parseInt(key, 10)].values.push({ 'x': i, 'y': value })

    x_min = (_.min(_.map(new_set, (xs) -> _.min(_.pluck(xs.values, 'x')))))
    x_max = (_.max(_.map(new_set, (xs) -> _.max(_.pluck(xs.values, 'x')))))
    x = d3.scale.linear()
        .domain([x_min, x_max])
        .range([0, width])

    y = d3.scale.linear()
        .domain([0, 1])
        .range([height, 0])

    # get the bars of the layout
    bars = d3.layout.stack().values((d) -> d.values)(new_set)

    ################################
    # graphics

    # area
    area = d3.svg.area()
        .x((d) -> x(d.x))
        .y0((d) -> y(d.y0))
        .y1((d) -> y(d.y0 + d.y))

    colour = d3.scale.category20()

    layers = chart.selectAll('.layer')
        .data(bars)

    layers.enter()
        .append('g')
        .attr('class', 'layer')

    layers.append('path')
        .attr('class', 'area')
        .attr('d', (d) -> area(d.values))
        .style('fill', (d) -> colour(d.name))

    x_axis = d3.svg.axis().scale(x).orient('bottom').tickValues(x for x in [x_min..x_max] by 100)
    y_axis = d3.svg.axis().scale(y).orient('left')

    if 'y_format' of params
        params.y_format(y_axis)

    create_axes x, y, width, height, margin, x_axis, y_axis, params.x_label, params.y_label, svg, chart

line_chart = (data, width, height, svg, params) ->
    # margins
    margin =
        'top': 20
        'bottom': 70
        'right': 140
        'left': 70

    [ width, height, chart ] = create_margin margin, width, height, svg

    # svg style sheet
    svg_style = svg.append('defs')
        .append('style')
        .attr('type','text/css')

    css_text = '<![CDATA[
                    text { font: 12px sans-serif; }
                    .line { fill: none; stroke-width: 1; }
                    .axis path, .axis line { fill: none; stroke: black; shape-rendering: crispEdges; }
                ]]>'

    svg_style.text css_text

    colour = d3.scale.category20()
    padding = 10

    if 'annotations' of params
        # create the legend symbols for the annotations
        anno_labels = _.unique (_.pluck params.annotations, 'label')

        # just append to data
        data.push { 'name': label, 'values': [], 'type': 'circle' } for label in anno_labels

    # legend
    svg.append('g')
        .attr('transform', 'translate(' + (padding + margin.left + width) + ',' + margin.top + ')')
        .attr('id', 'legend-box')

    legend_box = svg.select('#legend-box')

    rects = legend_box.selectAll('rect')
        .data(data)

    rects.enter()
        .append('rect')

    rects.attr('y', (d, i) -> (i * 1.2) + 'em')
        .attr('width', (d) -> if d.type is 'circle' then 4 else 10)
        .attr('height', (d) -> if d.type is 'circle' then 4 else 2)
        .style('fill', (d) -> colour(d.name))
        .filter((d) -> d.type is 'circle')
        .attr('rx', 4)
        .attr('ry', 4)
        .attr('x', 2)

    labels = legend_box.selectAll('text')
        .data(data)

    labels.enter()
        .append('text')

    labels.attr('y', (d, i) -> 0.4 + (i * 1.6) + 'em')
        .attr('x', 14)
        .text((d) -> d.name)

    x = d3.scale.linear()
        .domain([0, 1000])
        .range([0, width])

    # find y_max
    y_max = if 'y_max' of params then params.y_max else _.max(_.map(_.pluck(data, 'values'), _.max))
    y_min = if 'y_min' of params then params.y_min else _.min(_.map(_.pluck(data, 'values'), _.min))
    y = d3.scale.linear()
        .domain([y_max, y_min])
        .range([0, height])

    line = d3.svg.line()
        .x((d, i) -> x(i))
        .y((d, i) -> y(d))

    # each path
    paths = chart.selectAll('path')
        .data(data)

    paths.enter()
        .append('path')

    paths.attr('class', 'line')
        .attr('d', (d) -> line(d.values))
        .style('stroke', (d) -> colour(d.name))

    x_axis = d3.svg.axis().scale(x).orient('bottom')
    y_axis = d3.svg.axis().scale(y).orient('left')

    if 'y_format' of params
        params.y_format(y_axis)

    if 'annotations' of params
        annos = chart.selectAll('anno.circle')
            .data(params.annotations)

        annos.enter()
            .append('circle')
            .attr('cx', (d) -> d.x)

        annos.attr('class', 'anno')
            .attr('cx', (d) -> x(d.x))
            .attr('cy', (d) -> y(d.y))
            .attr('r', 2)
            .attr('fill', (d) -> colour(d.label))


    # put in the annotations if they exist
    create_axes x, y, width, height, margin, x_axis, y_axis, params.x_label, params.y_label, svg, chart

#'name': x,
#'path': path,
#'graphs_folder': path + '/graphs',
#'data_set': data_set,
#'arch': arch,
#'effect': effect,
#'seed': seed,
#'model': model
# marginalised

sum = (arr) -> arr.reduce ((a,b) -> a + b), 0

# extract different distributions (data_set, effect, seed)
do () ->
    create_dir 'distributions'
    distributions = { }

    for set in data_dirs
        name = set.data_set + '_' + set.effect + '_' + set.seed
        if not (name of distributions)
            params =
                'x_label': 'Position of the data stream'
                'y_label': 'Composition'
                'y_format': (axis) -> axis.ticks(10, '%')

            svg = build_chart (JSON.parse fs.readFileSync(set.path + '/distribution.json')), stacked_bar, params
            fs.writeFileSync ('distributions/' + name + '.svg'), svg
            distributions[name] = true

# validation average of these selected sets
validation_average = (sets) ->
    # average the validation error
    validation_sets = (d3.csv.parse fs.readFileSync(set.path + '/validation.csv', { 'encoding': 'utf-8' }) for set in sets)

    # sum across columns
    return (sum(parseFloat(v[i].error) for v in validation_sets) / validation_sets.length for i in [0..validation_sets[0].length-1])

# validation average of these selected sets
time_average = (sets) ->
    # average the validation error
    validation_sets = (d3.csv.parse fs.readFileSync(set.path + '/validation.csv', { 'encoding': 'utf-8' }) for set in sets)

    # sum across columns
    return (sum(parseFloat(v[i].time) - parseFloat(v[0].time) for v in validation_sets) / validation_sets.length for i in [0..validation_sets[0].length-1])

neurons_recon = (set) ->
    nr_data = d3.csv.parse fs.readFileSync(set.path + '/neurons_recon.csv', { 'encoding': 'utf-8' })
    return [ (parseFloat(v.neurons) for v in nr_data), (parseFloat(v.reconstruction) for v in nr_data) ]

model_name_mapper = (name) ->
    switch name
        when 'combined' then 'Standard'
        when 'adapting--policyPooler' then 'Pool-10000'
        when 'adapting--policyMDAE' then 'MDAE'
        when 'adapting--policyDiscreteRL' then 'DiscreteRL'
        when 'adapting--policyContinuousState' then 'ContRL'
        else name

# average error and time - data_set + effect averaged over all seeds
group_error_time = (data_set_func, err_dir_name, time_dir_name, y_err_label, y_time_label) ->
    create_dir err_dir_name
    create_dir time_dir_name

    for effective_data_set, v1 of (_.groupBy data_dirs, data_set_func)
        # group by marginalisation
        for marginalised, v2 of (_.groupBy v1, 'marginalised')
            for arch, v3 of (_.groupBy v2, 'arch')
                params =
                    'x_label': 'Position of the data stream'
                    'y_label': y_err_label
                    'y_format': (axis) -> axis.ticks(10, '%')
                    'y_min': 0
                    'y_max': 1

                error_data = ({ 'name': model_name_mapper(model), 'values': validation_average v4 } for model, v4 of (_.groupBy v3, 'model'))
                error_data = _.sortBy(error_data, (x) -> parseInt(x.name.split('=')[1], 10))

                svg = build_chart error_data, line_chart, params
                fs.writeFileSync (err_dir_name + '/' + [ effective_data_set, marginalised, arch ].join('_') + '.svg'), svg

                params =
                    'x_label': 'Position of the data stream'
                    'y_label': y_time_label

                time_data = ({ 'name': model_name_mapper(model), 'values': time_average v4 } for model, v4 of (_.groupBy v3, 'model'))
                svg = build_chart time_data, line_chart, params
                fs.writeFileSync (time_dir_name + '/' + [ effective_data_set, marginalised, arch ].join('_') + '.svg'), svg

# average over seed
group_error_time ((x) -> x.data_set + '_' + x.effect), 'average_err', 'average_time', 'Average error over all variations', 'Average time over all variations (s)'

# individual
group_error_time ((x) -> x.data_set + '_' + x.effect + '_' + x.seed), 'individual_err', 'individual_time', 'Error', 'Time (s)'

# build annotated
do () ->
    create_dir 'annotated'
    for set in data_dirs
        validation_data = { 'name': model_name_mapper(set.model), 'values': validation_average [set] }

        params =
            'x_label': 'Position of the data stream'
            'y_label': 'Error'
            'y_format': (axis) -> axis.ticks(10, '%')
            'y_min': 0
            'y_max': 1

        if fs.existsSync (set.path + '/actions.json')
            params.annotations = JSON.parse(fs.readFileSync(set.path + '/actions.json', { 'encoding': 'utf-8' }))
            params.annotations = ( { 'x': x, 'y': validation_data.values[x], 'label': params.annotations[x] } for x in [0..(validation_data.values.length-1)] )

        svg = build_chart [validation_data], line_chart, params
        fs.writeFileSync ('annotated/' + set.name + '.svg'), svg

# neuron balance
do () ->
    create_dir 'neurons'
    create_dir 'reconstruction'

    for set in data_dirs
        if fs.existsSync (set.path + '/neurons_recon.csv')
            [ neurons, reconstruction ] = neurons_recon set

            params =
                'x_label': 'Position of the data stream'
                'y_label': 'Neurons as percentage of initial value'
                'y_format': (axis) -> axis.ticks(10, '%')

            data = { 'name': model_name_mapper(set.model), 'values': neurons }

            if fs.existsSync (set.path + '/actions.json')
                params.annotations = JSON.parse(fs.readFileSync(set.path + '/actions.json', { 'encoding': 'utf-8' }))
                params.annotations = ( { 'x': x, 'y': data.values[x], 'label': params.annotations[x] } for x in [0..(data.values.length-1)] )

            svg = build_chart [data], line_chart, params
            fs.writeFileSync ('neurons/' + set.name + '.svg'), svg

            params =
                'x_label': 'Position of the data stream'
                'y_label': 'Reconstruction cost'
                'y_max': 500

            data = { 'name': model_name_mapper(set.model), 'values': reconstruction }

            if fs.existsSync (set.path + '/actions.json')
                params.annotations = JSON.parse(fs.readFileSync(set.path + '/actions.json', { 'encoding': 'utf-8' }))
                params.annotations = ( { 'x': x, 'y': data.values[x], 'label': params.annotations[x] } for x in [0..(data.values.length-1)] )

            svg = build_chart [data], line_chart, params
            fs.writeFileSync ('reconstruction/' + set.name + '.svg'), svg
