#### Libraries ###
library(shiny)
library(shinyalert)
library(shinyWidgets)
library(tidyverse)
library(reshape2)
library(gridExtra)
library(ggrepel)
library(png)
options(shiny.maxRequestSize = 10 * 1000 * 1024^2)

#### Settings ####
# graph parameters
pointsize <- 4

# shiny parameters
plot_wait_time <- 1000 # wait 1s until plot and table are generated

# search parameters
time_choices <- c(10, 30, 60, 120, 300, 600, 900)
dist_choices <- seq(0, 3, .5)
default_time <- 300
default_dist <- 1.5
alt_time <- 60
alt_dist <- 1


#### Building ####
background <- readPNG("../data-raw/background/image3195.png")
background_grob <- grid::rasterGrob(
  background,
  width = unit(1, "npc"),
  height = unit(1, "npc"),
  interpolate = TRUE
)
building_pl <- ggplot() +
  annotation_custom(
    background_grob,
    xmin = 0, xmax = 51, ymin = -0.02, ymax = 14.21
  ) +
  geom_rect(
    aes(xmin = 9.1, xmax = 11.3, ymin = 2.7, ymax = 3.7),
    fill = "yellow", color = NA, alpha = .25
  ) +
  geom_rect(
    aes(xmin = 9.1, xmax = 11.3, ymin = 3.71, ymax = 6.25),
    fill = "lightblue", color = NA, alpha = .25
  ) +
  xlim(0, 51) + # Set x-axis limits based on image extent
  ylim(-0.02, 14.21) + # Set y-axis limits based on image extent
  theme(panel.background = element_blank())
building_pl


#### Clinical data ####
clinic <- read.csv("../data-clean/clinical/tb_cases.csv")


#### Functions ####

euclidean <- function(x1, x2, y1, y2) {
  sqrt((x1 - x2)^2 + (y1 - y2)^2)
}

read_tracking <- function(file) {
  # tracking data
  tracks <- data.table::fread(
    file,
    select = c(
      "track_id",
      "time",
      "position_x", "position_y", "person_height",
      "near_entry", "in_tb_pat", "in_vitals_pat"
    ),
    integer64 = "numeric"
  ) %>%
    rename(
      x = position_x,
      y = position_y,
      height = person_height
    )

  # add mappings after preprocessing
  date <- as.Date(as.POSIXct(tracks$time[1] / 1000, origin = "1970-01-01"))
  mappings_csv <- paste0("../data-clean/tracking/linked/", date, ".csv")
  mappings <- read.csv(mappings_csv) %>%
    rename(
      new_track_id = track_id,
      track_id = raw_track_id
    )
  tracks <- left_join(tracks, mappings, by = "track_id")

  # existing links
  links_csv <- paste0("../data-clean/tracking/linked-tb/", date, ".csv")
  if (file.exists(links_csv)) {
    links <- read.csv(links_csv)
  } else {
    links <- tracks %>%
      group_by(new_track_id) %>%
      summarise(
        sum_tb = sum(c(NA, diff(time))[in_tb_pat], na.rm = TRUE) / 1000,
        sum_vit = sum(c(NA, diff(time))[in_vitals_pat], na.rm = TRUE) / 1000,
        entered = near_entry[1],
        exited = near_entry[n()],
        en_dist = euclidean(x[1], 10.4, y[1], 3.25),
        ex_dist = euclidean(x[n()], 10.4, y[n()], 3.25)
      ) %>%
      ungroup() %>%
      mutate(
        tb = case_when(
          (sum_tb > 180) & (sum_vit > 30) ~ TRUE,
          entered & exited & (sum_tb > 60) & (sum_vit > 30) ~ TRUE,
          !entered & exited & (sum_tb > 10) & (en_dist < 3) ~ TRUE,
          entered & !exited & (sum_tb > 10) & (ex_dist < 3) ~ TRUE,
          .default = FALSE
        ),
        sure_tb = case_when(
          (sum_tb > 180) & (sum_vit > 30) ~ TRUE,
          .default = FALSE
        ),
        seen = FALSE
      ) %>%
      left_join(mappings, by = "new_track_id") %>%
      dplyr::select(track_id, new_track_id, tb, sure_tb, seen)
  }

  # merge
  tracks <- left_join(
    tracks,
    links %>% dplyr::select(-new_track_id),
    by = "track_id"
  )

  return(tracks)
}

duration_min_sec <- function(time, subset = NULL) {
  if (is.null(subset)) {
    subset <- rep(TRUE, length(time))
  }
  dt <- c(NA, diff(time))
  total <- sum(dt[subset], na.rm = TRUE) / 1000
  minutes <- floor(total / 60)
  seconds <- round(total - minutes * 60)
  paste0(minutes, "min ", seconds, "sec")
}

update_datetime <- function(df, id) {
  as.character(as.POSIXct(
    df$time[df$new_track_id == id][1] / 1000,
    origin = "1970-01-01"
  ))
}

update_ids <- function(values) {
  values$ids <- as.integer(
    values$dat$new_track_id[values$dat$tb]
  )
  values$ids_pu <- as.integer(unique(
    values$dat$new_track_id[
      values$dat$tb &
        !values$dat$seen &
        !values$dat$sure_tb
    ]
  ))
  values$ids_pc <- as.integer(unique(
    values$dat$new_track_id[
      values$dat$tb &
        values$dat$seen &
        !values$dat$sure_tb
    ]
  ))
  values$ids_du <- as.integer(unique(
    values$dat$new_track_id[
      values$dat$tb &
        !values$dat$seen &
        values$dat$sure_tb
    ]
  ))
  values$ids_dc <- as.integer(unique(
    values$dat$new_track_id[
      values$dat$tb &
        values$dat$seen &
        values$dat$sure_tb
    ]
  ))
}

display_id_counts <- function(df) {
  if (is.null(df)) {
    return("")
  } else {
    n_pu <- n_distinct(df$new_track_id[df$tb & !df$seen & !df$sure_tb])
    n_pc <- n_distinct(df$new_track_id[df$tb & df$seen & !df$sure_tb])
    n_p <- n_pu + n_pc
    n_du <- n_distinct(df$new_track_id[df$tb & !df$seen & df$sure_tb])
    n_dc <- n_distinct(df$new_track_id[df$tb & df$seen & df$sure_tb])
    n_d <- n_du + n_dc
    counts <- paste0(
      "Maybe: ", n_p, " (", n_pc, " checked), ",
      "Sure: ", n_d, " (", n_dc, " checked)"
    )
    return(counts)
  }
}

update_linkage <- function(df, file_path) {
  write.csv(
    x = df %>%
      group_by(track_id, new_track_id) %>%
      slice(1) %>%
      ungroup() %>%
      dplyr::select(track_id, new_track_id, tb, sure_tb, seen),
    file = file_path,
    row.names = FALSE
  )
}

filter_tracks <- function(df, id, direction = 1) {
  if (direction == 1) {
    # current id
    df_i <- df %>%
      filter(new_track_id == id) %>%
      slice(n())

    # pre time filter conditions
    earliest <- df_i$time[1] - 3000
    latest <- df_i$time[1] + max(time_choices) * 1000

    # other potential ids
    df_other <- df %>%
      filter(new_track_id > id) %>%
      group_by(new_track_id) %>%
      slice(1) %>%
      ungroup() %>%
      filter(between(time, earliest, latest))
  } else {
    # current id
    df_i <- df %>%
      filter(new_track_id == id) %>%
      slice(1)

    # pre time filter conditions
    latest <- df_i$time[1] + 3000
    earliest <- df_i$time[1] - max(time_choices) * 1000

    # other potential ids
    df_other <- df %>%
      filter(new_track_id < id) %>%
      group_by(new_track_id) %>%
      slice(n()) %>%
      ungroup() %>%
      filter(between(time, earliest, latest))
  }

  # merge
  df_other <- base::merge(
    df_other, df_i,
    suffixes = c("_other", "_i"),
    by = NULL
  )

  return(df_other)
}

compute_features <- function(df_other, direction) {
  if (direction == 1) {
    df_other %>%
      mutate(
        timediff_raw = (time_other - time_i) / 1000,
        timediff = abs(timediff_raw),
        distance = euclidean(x_other, x_i, y_other, y_i)
      )
  } else {
    df_other %>%
      mutate(
        timediff_raw = (time_i - time_other) / 1000,
        timediff = abs(timediff_raw),
        distance = euclidean(x_other, x_i, y_other, y_i)
      )
  }
}


plot_ids <- function(pl, df_i, df_pos, df_alt, df_pos_feat) {
  if (!is.null(df_i)) {
    if (nrow(df_i) > 0) {
      # df_i first and last track
      df_i_f <- slice(df_i, 1)
      df_i_l <- slice(df_i, n())

      # df_i plot
      pl <- pl +
        geom_path(
          data = df_i,
          mapping = aes(x = x, y = y),
          color = "black"
        ) +
        geom_point(
          data = df_i_f,
          mapping = aes(x = x, y = y),
          shape = 1, color = "black", size = pointsize
        ) +
        geom_point(
          data = df_i_l,
          mapping = aes(x = x, y = y),
          shape = 13, color = "black", size = pointsize
        )
    }
  }

  if (!is.null(df_pos)) {
    if (nrow(df_pos) > 0) {
      pid_order <- df_pos_feat$new_track_id_other[order(df_pos_feat$timediff_raw)]

      df_pos <- df_pos %>%
        mutate(new_track_id = factor(new_track_id, levels = pid_order))

      df_pos_f <- df_pos %>%
        group_by(new_track_id) %>%
        slice(1)
      df_pos_l <- df_pos %>%
        group_by(new_track_id) %>%
        slice(n())

      pl <- pl +
        geom_path(
          data = df_pos,
          mapping = aes(x = x, y = y, color = new_track_id)
        ) +
        geom_point(
          data = df_pos_f,
          mapping = aes(x = x, y = y, color = new_track_id),
          shape = 1, fill = "white", size = pointsize
        ) +
        geom_point(
          data = df_pos_l,
          mapping = aes(x = x, y = y, color = new_track_id),
          shape = 13, fill = "white", size = pointsize
        ) +
        theme(legend.position = "none", legend.title = element_blank())
    }
  }

  if (!is.null(df_alt)) {
    if (nrow(df_alt) > 0) {
      df_alt_f <- df_alt %>%
        group_by(new_track_id) %>%
        slice(1)
      df_alt_l <- df_alt %>%
        group_by(new_track_id) %>%
        slice(n())

      pl <- pl +
        geom_path(
          data = df_alt,
          mapping = aes(x = x, y = y, group = new_track_id),
          alpha = .5
        ) +
        geom_text_repel(
          data = df_alt_l,
          mapping = aes(x = x, y = y, group = new_track_id, label = new_track_id),
          alpha = .66, size = 10 / cm(1)
        ) +
        geom_point(
          data = df_alt_f,
          mapping = aes(x = x, y = y, group = new_track_id),
          shape = 1, fill = "white", size = pointsize, alpha = .5
        ) +
        geom_point(
          data = df_alt_l,
          mapping = aes(x = x, y = y, group = new_track_id),
          shape = 13, fill = "white", size = pointsize, alpha = .5
        )
    }
  }

  return(pl)
}

table_ids <- function(df_i, df_pos, direction) {
  if (is.null(df_i)) {
    return(NULL)
  }
  if (is.null(df_pos)) {
    return(NULL)
  }
  if (nrow(df_i) == 0) {
    return(NULL)
  }
  if (nrow(df_pos) == 0) {
    return(NULL)
  }

  # last values from ID
  if (direction == 1) {
    df_i <- df_i %>%
      mutate(mean_height = mean(height)) %>%
      tail(1)
  } else {
    df_i <- df_i %>%
      mutate(mean_height = mean(height)) %>%
      head(1)
  }

  # first values from possible links
  df_pos_1 <- df_pos %>%
    group_by(new_track_id) %>%
    summarize(
      dur = duration_min_sec(time),
      mean_height = mean(height)
    ) %>%
    ungroup()
  if (direction == 1) {
    df_pos_2 <- df_pos %>%
      group_by(new_track_id) %>%
      slice(1) %>%
      ungroup()
  } else {
    df_pos_2 <- df_pos %>%
      group_by(new_track_id) %>%
      slice(n()) %>%
      ungroup()
  }

  df_pos <- left_join(df_pos_1, df_pos_2, by = "new_track_id")


  # features
  df_feat <- base::merge(df_pos, df_i, suffixes = c("_pos", "_i"), by = NULL)

  if (direction == 1) {
    df_feat <- df_feat %>%
      mutate(
        timediff = (time_pos - time_i) / 1000,
        distance = euclidean(x_pos, x_i, y_pos, y_i),
        last_heightdiff = height_pos - height_i,
        mean_heightdiff = mean_height_pos - mean_height_i
      )
  } else {
    df_feat <- df_feat %>%
      mutate(
        timediff = (time_i - time_pos) / 1000,
        distance = euclidean(x_pos, x_i, y_pos, y_i),
        last_heightdiff = height_i - height_pos,
        mean_heightdiff = mean_height_i - mean_height_pos
      )
  }

  df_feat <- df_feat %>%
    mutate(
      across(contains("height"), ~ format(round(.x, 2), nsmall = 2)),
      distance = format(round(distance, 1), nsmall = 1)
    ) %>%
    mutate(
      last_height_comb = paste0(
        last_heightdiff, " (",
        height_pos, ", ",
        height_i, ")"
      ),
      mean_height_comb = paste0(
        mean_heightdiff, " (",
        mean_height_pos, ", ",
        mean_height_i, ")"
      )
    ) %>%
    arrange(timediff) %>%
    dplyr::select(
      new_track_id_i,
      new_track_id_pos,
      last_height_comb,
      mean_height_comb,
      dur,
      timediff,
      distance
    ) %>%
    set_names(
      "ID",
      "Links",
      "Last heightdiff [cm]",
      "Mean heightdiff [cm]",
      "Duration [min]",
      "Timediff [sec]",
      "Distance [m]"
    ) %>%
    mutate_all(as.character)

  n_pos_ids <- nrow(df_feat)

  if (n_pos_ids > 0) {
    for (i in 1:n_pos_ids) {
      col <- scales::hue_pal()(n_pos_ids)[i]
      for (k in 2:6) {
        df_feat[i, k] <- paste0('<span style="color:', col, '">', df_feat[i, k], "</span>")
      }
    }
  }

  return(df_feat)
}


#### Shiny UI ####
ui <- fluidPage(
  sidebarLayout(
    sidebarPanel(
      tags$style(
        HTML(".bold-prefix { font-weight: bold; margin-right: 5px; }")
      ),
      fileInput("file", "Load file"),
      div(
        tags$span("Date and time:", class = "bold-prefix"),
        textOutput("date"),
        style = "display: flex;"
      ),
      div(
        tags$span("Trackings:", class = "bold-prefix"),
        textOutput("tracking_ids"),
        style = "display: flex;"
      ),
      div(
        tags$span("Clinical records:", class = "bold-prefix"),
        textOutput("clinical_ids"),
        style = "display: flex;"
      ),
      br(),
      selectizeInput(
        "id", "Currently selected ID",
        choices = -1,
        selected = -1,
        multiple = FALSE,
        options = list(maxItems = 1)
      ),
      radioButtons(
        "show_ids", "Show",
        choices = list(
          "Maybe" = 1,
          "Maybe (checked)" = 2,
          "Sure" = 3,
          "Sure (checked)" = 4
        ),
        selected = 3,
        inline = TRUE
      ),
      actionButton("next_id", "Next ID"),
      br(),
      br(),
      radioButtons(
        "direction", "Matching direction",
        choices = list("Forward" = 1, "Backward" = 2),
        inline = TRUE
      ),
      div(
        tags$span("ID:", class = "bold-prefix"),
        textOutput("id_info"),
        style = "display: flex;"
      ),
      br(),
      sliderTextInput("time", "Time",
        from_min = min(time_choices),
        to_max = max(time_choices),
        selected = default_time,
        choices = time_choices,
        grid = TRUE,
        post = "sec"
      ),
      sliderTextInput("distance", "Distance",
        from_min = min(dist_choices),
        to_max = max(dist_choices),
        selected = default_dist,
        choices = dist_choices,
        grid = TRUE,
        post = "m"
      ),
      selectizeInput(
        "alt_id", "Show alternatives for ID",
        choices = -1,
        selected = -1,
        options = list(maxItems = 1)
      ),
      selectizeInput(
        "pos_id", "Link with ID",
        choices = -1,
        selected = -1,
        options = list(maxItems = 1)
      ),
      actionButton("link", "Link IDs"),
      actionButton("unlink_last", "Unlink last ID"),
      actionButton("unlink_first", "Unlink first ID"),
      br(),
      br(),
      selectInput(
        "is_tb", "Is TB patient?",
        choices = c("Not TB", "Sure TB")
      ),
      actionButton("finish", "Finish"),
      br(),
      br(),
      div(
        tags$span("Saved path-file:", class = "bold-prefix"),
        textOutput("saveto"),
        style = "display: flex;"
      ),
    ),
    mainPanel(
      plotOutput("clinic", inline = TRUE),
      br(),
      tableOutput("links_table")
    )
  )
)



#### Shiny Sever ####
server <- function(input, output, session) {
  #### Load data ####
  values <- reactiveValues(
    dat = NULL, # data
    ids = NULL, # IDs
    ids_pu = NULL, # possible checked IDs
    ids_pc = NULL, # possible unchecked IDs
    ids_du = NULL, # definitive unchecked IDs
    ids_dc = NULL, # definitive checked IDS
    current_id_type = NULL,
    previous_id_type = TRUE,
    update_id = TRUE,
    dat_i = NULL, # data of selected id
    dat_o = NULL, # data of other possible id links
    dat_o_feat = NULL, # feature data of other
    dat_os_feat = NULL, # subset of feature data
    dat_a = NULL # data of alternatives for selected other possible id link
  )
  observeEvent(input$file, {
    # get file
    file <- input$file

    # read tracking data
    values$dat <- read_tracking(file$datapath)

    # date
    date <- as.Date(as.POSIXct(
      values$dat$time[1] / 1000,
      origin = "1970-01-01"
    ))

    # update ID subsets
    update_ids(values)

    # update patient ID selection
    values$current_id_type <- 3
    if (length(values$ids_du) > 0) {
      first_id <- values$ids_du[1]
    } else {
      first_id <- values$ids_dc[1]
    }

    updateSelectizeInput(
      session, "id",
      choices = values$ids,
      selected = first_id,
      server = TRUE
    )

    # initial counts
    output$tracking_ids <- renderText({
      display_id_counts(values$dat)
    })

    # clinic ID count
    output$clinical_ids <- renderText({
      n_distinct(clinic$clinic_id[clinic$date == date])
    })

    # directory to save file
    values$save_file <- paste0(
      "../data-clean/tracking/linked-tb/", date, ".csv"
    )
    output$saveto <- renderText({
      values$save_file
    })
  })

  #### Select ID ####
  # show ID subset
  observeEvent(input$show_ids, {
    if (input$show_ids == 1) {
      next_selected <- values$ids_pu[1]
    } else if (input$show_ids == 2) {
      next_selected <- values$ids_pc[1]
    } else if (input$show_ids == 3) {
      next_selected <- values$ids_du[1]
    } else {
      next_selected <- values$ids_dc[1]
    }
    if (values$update_id) {
      updateSelectizeInput(
        session,
        inputId = "id",
        choices = values$ids,
        selected = next_selected,
        server = TRUE
      )
    }
  })

  # select next ID
  observeEvent(input$next_id, {
    req(input$id)
    id <- as.integer(input$id)
    if (input$show_ids == 1) {
      values$ids_pu <- values$ids_pu[values$ids_pu != id]
      values$ids_pc <- c(values$ids_pc, id)
      values$dat$seen[values$dat$new_track_id == id] <- TRUE
      next_selected <- values$ids_pu[1]
      update_linkage(values$dat, values$save_file)
    } else if (input$show_ids == 2) {
      next_selected <- values$ids_pc[values$ids_pc > id][1]
    } else if (input$show_ids == 3) {
      values$ids_du <- values$ids_du[values$ids_du != id]
      values$ids_dc <- c(values$ids_dc, id)
      values$dat$seen[values$dat$new_track_id == id] <- TRUE
      next_selected <- values$ids_du[1]
      update_linkage(values$dat, values$save_file)
    } else {
      next_selected <- values$ids_dc[values$ids_dc > id][1]
    }
    updateSelectizeInput(
      session,
      inputId = "id",
      choices = values$ids,
      selected = next_selected,
      server = TRUE
    )
  })

  #### Data ####

  # ID data and
  observeEvent(input$id, {
    if (!is.null(values$dat)) {
      # get id
      id <- as.integer(input$id)
      if (is.na(id)) {
        return()
      }

      # filter id data
      values$dat_i <- values$dat[values$dat$new_track_id == id, ]

      # update inputs
      # ID type should be silently updated for manually selected IDs
      values$previous_id_type <- values$current_id_type
      values$current_id_type <- case_when(
        values$dat_i$tb[1] & !values$dat_i$seen[1] & !values$dat_i$sure_tb[1] ~ 1,
        values$dat_i$tb[1] & values$dat_i$seen[1] & !values$dat_i$sure_tb[1] ~ 2,
        values$dat_i$tb[1] & !values$dat_i$seen[1] & values$dat_i$sure_tb[1] ~ 3,
        values$dat_i$tb[1] & values$dat_i$seen[1] & values$dat_i$sure_tb[1] ~ 4
      )
      if (values$previous_id_type == values$current_id_type) {
        values$update_id <- TRUE
      } else {
        values$update_id <- FALSE
        updateRadioButtons(
          session,
          "show_ids",
          selected = values$current_id_type
        )
      }
      updateRadioButtons(session, "direction", selected = 1)
      updateSliderTextInput(session, "time", selected = default_time)
      updateSliderTextInput(session, "distance", selected = default_dist)

      # update info
      values$i_links <- n_distinct(values$dat_i$track_id) - 1
      output$id_info <- renderText({
        paste0(
          duration_min_sec(values$dat_i$time),
          " (TB: ", duration_min_sec(values$dat_i$time, values$dat_i$in_tb_pat),
          ", Vitals: ", duration_min_sec(values$dat_i$time, values$dat_i$in_vitals_pat),
          ", Links: ", values$i_links, ")"
        )
      })
      output$date <- renderText({
        update_datetime(values$dat, id)
      })
      output$tracking_ids <- renderText({
        display_id_counts(values$dat)
      })

      # compute features for others
      values$dat_o_feat <- filter_tracks(values$dat, id, input$direction)
      values$dat_o_feat <- compute_features(values$dat_o_feat, input$direction)
    }
  })

  observeEvent(input$direction, {
    if (
      !is.null(values$dat) &
        !is.null(values$dat_i) &
        !is.null(values$dat_o_feat)
    ) {
      id <- as.integer(input$id)
      # compute features for others
      values$dat_o_feat <- filter_tracks(values$dat, id, input$direction)
      values$dat_o_feat <- compute_features(values$dat_o_feat, input$direction)
    }
  })

  # possible link ID data
  observe({
    # get possible links
    if (!is.null(values$dat_o_feat)) {
      id <- as.integer(input$id)
      if (nrow(values$dat_o_feat) > 0) {
        values$dat_os_feat <- values$dat_o_feat %>%
          filter(
            timediff <= input$time,
            distance <= input$distance
          )
        if (nrow(values$dat_os_feat) > 0) {
          pos_ids <- values$dat_os_feat$new_track_id_other
          values$dat_o <- filter(values$dat, new_track_id %in% pos_ids)
        } else {
          values$dat_o <- NULL
        }
      } else {
        values$dat_o <- NULL
      }
    } else {
      values$dat_o <- NULL
    }

    # update selection
    if (!is.null(values$dat_o)) {
      pos_ids <- pos_ids[order(values$dat_os_feat$timediff_raw)]
      updateSelectizeInput(session, inputId = "pos_id", choices = pos_ids)
      updateSelectizeInput(session, inputId = "alt_id", choices = pos_ids)
    } else {
      updateSelectizeInput(session, inputId = "pos_id", choices = -1)
      updateSelectizeInput(session, inputId = "alt_id", choices = -1)
    }
  })

  # alternatives data
  observeEvent(input$alt_id, {
    if (!is.null(values$dat_o)) {
      alt_id <- as.integer(input$alt_id)
      id <- as.integer(input$id)
      dat_o_i <- filter(values$dat, new_track_id == alt_id)
      values$dat_a_feat <- filter_tracks(
        values$dat,
        alt_id,
        ifelse(input$direction == 1, 2, 1)
      ) %>%
        filter(new_track_id_other != id)
      if (nrow(values$dat_a_feat) > 0) {
        values$dat_a_feat <- compute_features(
          values$dat_a_feat,
          ifelse(input$direction == 1, 2, 1)
        )
        dat_as_feat <- values$dat_a_feat %>%
          filter(
            timediff <= alt_time,
            distance <= alt_dist
          )
        alt_ids <- dat_as_feat$new_track_id_other
        values$dat_a <- filter(values$dat, new_track_id %in% alt_ids)
      } else {
        values$dat_a <- NULL
      }
    } else {
      values$dat_a <- NULL
    }
  })


  #### New link ####
  observeEvent(input$link, {
    # make link
    id <- as.integer(input$id)
    pos_id <- as.integer(input$pos_id)
    if (pos_id == -1) {
      shinyalert(
        "Error: -1",
        name_linkage_success,
        type = "success",
        timer = 1000
      )
    } else {
      linked_ids <- c(id, pos_id)
      new_id <- max(linked_ids)
      name_linkage_success <- paste("ID", id, "linked to", pos_id, ".")
      shinyalert(
        "Success",
        name_linkage_success,
        type = "success",
        timer = 1000
      )
      values$dat$new_track_id[values$dat$new_track_id %in% linked_ids] <- new_id
      values$dat$seen[values$dat$new_track_id == new_id] <- FALSE
      values$dat$tb[values$dat$new_track_id == new_id] <- TRUE
      values$dat$sure_tb[values$dat$new_track_id == new_id] <- any(values$dat$sure_tb[values$dat$new_track_id == new_id])
      update_linkage(values$dat, values$save_file)

      # update selection
      update_ids(values)
      updateSelectizeInput(
        inputId = "id",
        choices = values$ids,
        selected = new_id,
        server = TRUE
      )
    }
  })

  #### Un-link ####
  observeEvent(input$unlink_last, {
    # unlink last ID
    last_id <- as.integer(input$id)
    old_id <- tail(sort(unique(
      values$dat$track_id[values$dat$new_track_id == last_id]
    )), 2)[1]
    shinyalert(
      "Info",
      paste("Unlinking ID", last_id, "from", old_id),
      type = "info",
      timer = 1000
    )
    values$dat$new_track_id[values$dat$new_track_id == last_id & values$dat$track_id != last_id] <- old_id
    values$dat$seen[values$dat$new_track_id %in% c(last_id, old_id)] <- FALSE
    update_linkage(values$dat, values$save_file)

    # update selection
    update_ids(values)
    updateSelectizeInput(
      inputId = "id",
      choices = values$ids,
      selected = old_id,
      server = TRUE
    )
  })

  observeEvent(input$unlink_first, {
    # unlink first ID
    last_id <- as.integer(input$id)
    old_id <- sort(unique(
      values$dat$track_id[values$dat$new_track_id == last_id]
    ))[1]
    values$dat$new_track_id[values$dat$track_id == old_id] <- old_id
    values$dat$seen[values$dat$track_id %in% c(last_id, old_id)] <- FALSE
    update_linkage(values$dat, values$save_file)
    shinyalert(
      "Info",
      paste("Unlinking ID", last_id, "from", old_id),
      type = "info",
      timer = 1000
    )

    # update selection
    update_ids(values)
    updateSelectizeInput(
      inputId = "id",
      choices = values$ids,
      selected = last_id,
      server = TRUE
    )
  })


  #### End track ####
  observeEvent(input$finish, {
    req(input$id)
    req(input$show_ids)
    # update mapping
    id <- as.integer(input$id)
    is_tb <- (input$is_tb == "Sure TB")
    values$dat$sure_tb[values$dat$new_track_id == id] <- is_tb
    values$dat$tb[values$dat$new_track_id == id] <- is_tb
    values$dat$seen[values$dat$new_track_id == id] <- TRUE
    update_linkage(values$dat, values$save_file)
    shinyalert(
      "Done.",
      paste("ID", input$id, "is", input$is_tb),
      type = "info",
      timer = 1000
    )
    update_ids(values)
    if (is_tb) {
      next_selected <- id
    } else {
      if (length(values$ids_du) > 0) {
        next_selected <- values$ids_du[1]
      } else {
        next_selected <- values$ids_dc[1]
      }
    }
    updateSelectizeInput(
      inputId = "id",
      choices = values$ids,
      selected = next_selected,
      server = TRUE
    )
  })


  #### Plot ####
  output$clinic <- renderPlot(
    {
      plot_ids(
        building_pl,
        values$dat_i,
        values$dat_o,
        values$dat_a,
        values$dat_os_feat
      )
    },
    height = 800,
    width = 1200
  )

  #### Table ####
  output$links_table <- renderTable(
    {
      table_ids(values$dat_i, values$dat_o, input$direction)
    },
    sanitize.text.function = function(x) x
  )
}





#### Shiny App ####
shinyApp(ui = ui, server = server)
