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
min_alt_time <- 60
min_alt_dist <- 1


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

standing_height <- function(x, k = 20) {
  n <- length(x)
  if (n < k) {
    return(max(x))
  } else {
    return(max(zoo::rollmean(x, k)))
  }
}

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
      "tag",
      "near_entry", "in_check", "in_check_tb", "in_sputum"
    ),
    integer64 = "numeric"
  ) %>%
    rename(
      x = position_x,
      y = position_y,
      height = person_height
    )

  # add mappings after preprocessing
  file_date <- as.Date(
    as.POSIXct(tracks$time[1] / 1000, origin = "1970-01-01")
  )
  mappings_csv <- paste0("../data-clean/tracking/linked/", file_date, ".csv")
  mappings <- read.csv(mappings_csv) %>%
    rename(
      new_track_id = track_id,
      track_id = raw_track_id
    )

  # existing links
  links_csv <- paste0("../data-clean/tracking/linked-tb/", file_date, ".csv")
  if (file.exists(links_csv)) {
    links <- read.csv(links_csv)
    tracks <- left_join(
      tracks,
      links,
      by = "track_id"
    )
  } else {
    tracks <- left_join(tracks, mappings, by = "track_id")
    clin_sub <- subset(clinic, date == file_date)
    last_clin_time <- max(as.POSIXct(clin_sub$start_time))
    links <- tracks %>%
      group_by(new_track_id) %>%
      summarise(
        tag = any(tag),
        start_time = as.POSIXct(min(time) / 1000, origin = "1970-01-01"),
        time_tb = sum(c(NA, diff(time))[in_check_tb], na.rm = TRUE) / 1000,
        sputum = any(in_sputum),
        missing_start = in_check_tb[1],
        missing_end = in_check_tb[n()]
      ) %>%
      ungroup() %>%
      mutate(
        category = case_when(
          between(time_tb, 60, 600) ~ "possible",
          between(time_tb, 30, 59) ~ "maybe",
          between(time_tb, 601, 1200) ~ "maybe",
          !between(time_tb, 60, 300) & missing_start ~ "maybe",
          !between(time_tb, 60, 300) & missing_end ~ "maybe",
          .default = "not TB"
        ),
        category = ifelse(sputum, "sure", category),
        category = ifelse(tag, "not TB", category),
        category = ifelse(start_time > last_clin_time, "not TB", category)
      ) %>%
      left_join(mappings, by = "new_track_id") %>%
      dplyr::select(track_id, new_track_id, category)
    tracks <- left_join(
      tracks,
      links %>% dplyr::select(-new_track_id),
      by = "track_id"
    )
  }

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
  paste0(minutes, "m ", seconds, "s")
}

update_datetime <- function(values) {
  as.character(as.POSIXct(
    values$dat_i$time[1] / 1000,
    origin = "1970-01-01"
  ))
}

update_ids <- function(values) {
  values$ids_pu <- as.integer(unique(
    values$dat$new_track_id[values$dat$category == "maybe"]
  ))
  values$ids_pc <- as.integer(unique(
    values$dat$new_track_id[values$dat$category == "possible"]
  ))
  values$ids_du <- as.integer(unique(
    values$dat$new_track_id[values$dat$category == "likely"]
  ))
  values$ids_dc <- as.integer(unique(
    values$dat$new_track_id[values$dat$category == "sure"]
  ))
  values$ids_sp <- as.integer(unique(
    values$dat$new_track_id[values$dat$category == "sputum only"]
  ))
}

get_next_id <- function(x, id) {
  n <- length(x)
  next_id <- which(x == id) + 1
  if (next_id > n) {
    return(x[1])
  } else {
    return(x[next_id])
  }
}

get_prev_id <- function(x, id) {
  n <- length(x)
  prev_id <- which(x == id) - 1
  if (prev_id <= 0) {
    return(x[n])
  } else {
    return(x[prev_id])
  }
}

update_id_selection <- function(session, values) {
  cat <- values$dat$category[values$dat$new_track_id == values$select_id][1]
  if (cat == "maybe") {
    show <- 1
  } else if (cat == "possible") {
    show <- 2
  } else if (cat == "likely") {
    show <- 3
  } else if (cat == "sure") {
    show <- 4
  } else {
    show <- 5
  }
  if (show == 1) {
    values$prev_id <- get_prev_id(values$ids_pu, values$select_id)
    values$next_id <- get_next_id(values$ids_pu, values$select_id)
    updateSelectizeInput(
      session, "id",
      choices = values$ids_pu,
      selected = values$select_id,
      server = TRUE
    )
  } else if (show == 2) {
    values$prev_id <- get_prev_id(values$ids_pc, values$select_id)
    values$next_id <- get_next_id(values$ids_pc, values$select_id)
    updateSelectizeInput(
      session, "id",
      choices = values$ids_pc,
      selected = values$select_id,
      server = TRUE
    )
  } else if (show == 3) {
    values$prev_id <- get_prev_id(values$ids_du, values$select_id)
    values$next_id <- get_next_id(values$ids_du, values$select_id)
    updateSelectizeInput(
      session, "id",
      choices = values$ids_du,
      selected = values$select_id,
      server = TRUE
    )
  } else if (show == 4) {
    values$prev_id <- get_prev_id(values$ids_dc, values$select_id)
    values$next_id <- get_next_id(values$ids_dc, values$select_id)
    updateSelectizeInput(
      session, "id",
      choices = values$ids_dc,
      selected = values$select_id,
      server = TRUE
    )
  } else {
    values$prev_id <- get_prev_id(values$ids_sp, values$select_id)
    values$next_id <- get_next_id(values$ids_sp, values$select_id)
    updateSelectizeInput(
      session, "id",
      choices = values$ids_sp,
      selected = values$select_id,
      server = TRUE
    )
  }
}

update_info <- function(values) {
  if (is.na(values$dat_i$category[1])) {
    return("-")
  }
  cat <- values$dat_i$category[1]
  if (cat == "maybe") {
    show <- 1
  } else if (cat == "possible") {
    show <- 2
  } else if (cat == "likely") {
    show <- 3
  } else if (cat == "sure") {
    show <- 4
  } else {
    show <- 5
  }
  if (cat == "maybe") {
    n <- length(values$ids_pu)
    i <- which(values$ids_pu == values$dat_i$new_track_id[1])
  } else if (cat == "possible") {
    n <- length(values$ids_pc)
    i <- which(values$ids_pc == values$dat_i$new_track_id[1])
  } else if (cat == "likely") {
    n <- length(values$ids_du)
    i <- which(values$ids_du == values$dat_i$new_track_id[1])
  } else if (cat == "sure") {
    n <- length(values$ids_dc)
    i <- which(values$ids_dc == values$dat_i$new_track_id[1])
  } else {
    n <- length(values$ids_sp)
    i <- which(values$ids_sp == values$dat_i$new_track_id[1])
  }
  links <- n_distinct(values$dat_i$track_id) - 1
  info <- paste0(
    i, " of ", n, " [", cat, "] ",
    "with ", links, " [links]"
  )
  return(info)
}

update_time <- function(values) {
  time_1 <- duration_min_sec(values$dat_i$time)
  time_2 <- duration_min_sec(values$dat_i$time, values$dat_i$in_check_tb)
  time_3 <- duration_min_sec(values$dat_i$time, values$dat_i$in_check)
  time_info <- paste0(
    time_1, " [Total], ",
    time_2, " [TB], ",
    time_3, " [Check]"
  )
  return(time_info)
}

update_height <- function(values, direction) {
  if (is.na(values$dat_i$height[1])) {
    return("-")
  }
  mean_height <- round(median(values$dat_i$height), 2)
  max_height <- round(standing_height(values$dat_i$height), 2)
  if (direction == 1) {
    last_height <- round(tail(values$dat_i$height, 1), 2)
  } else {
    last_height <- round(values$dat_i$height[1], 2)
  }
  height_info <- paste0(
    last_height, " [Last], ",
    mean_height, " [Sit], ",
    max_height, " [Stand]"
  )
  return(height_info)
}

update_counts <- function(values) {
  if (is.null(values$dat)) {
    return("")
  } else {
    n_pu <- length(values$ids_pu)
    n_pc <- length(values$ids_pc)
    n_du <- length(values$ids_du)
    n_dc <- length(values$ids_dc)
    n_sp <- length(values$ids_sp)
    counts <- paste0(
      "Ma: ", n_pu, ", Po: ", n_pc,
      ", Li: ", n_du, " Su: ", n_dc,
      ", Sp: ", n_sp
    )
    return(counts)
  }
}

update_linkage <- function(values) {
  write.csv(
    x = values$dat %>%
      group_by(track_id, new_track_id) %>%
      slice(1) %>%
      ungroup() %>%
      dplyr::select(track_id, new_track_id, category),
    file = values$save_file,
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
      filter(!tag) %>%
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
  df_i_height <- df_i %>%
    group_by(new_track_id) %>%
    summarise(
      mean_height = median(height),
      max_height = standing_height(height)
    ) %>%
    ungroup()
  if (direction == 1) {
    df_i <- df_i %>%
      tail(1)
  } else {
    df_i <- df_i %>%
      head(1)
  }
  df_i <- left_join(df_i, df_i_height, by = "new_track_id")


  # first values from possible links
  df_pos_1 <- df_pos %>%
    group_by(new_track_id) %>%
    summarize(
      dur = duration_min_sec(time),
      mean_height = median(height),
      max_height = standing_height(height)
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
        distance = euclidean(x_pos, x_i, y_pos, y_i)
      )
  } else {
    df_feat <- df_feat %>%
      mutate(
        timediff = (time_i - time_pos) / 1000,
        distance = euclidean(x_pos, x_i, y_pos, y_i)
      )
  }

  df_feat <- df_feat %>%
    mutate(
      last_heightdiff = height_pos - height_i,
      mean_heightdiff = mean_height_pos - mean_height_i,
      max_heightdiff = max_height_pos - max_height_i,
      across(contains("height"), ~ format(round(.x, 2), nsmall = 2)),
      last_heightdiff = paste0(height_pos, " (", last_heightdiff, ")"),
      mean_heightdiff = paste0(mean_height_pos, " (", mean_heightdiff, ")"),
      max_heightdiff = paste0(max_height_pos, " (", max_heightdiff, ")"),
      distance = format(round(distance, 1), nsmall = 1),
      timediff = round(timediff)
    ) %>%
    arrange(timediff) %>%
    dplyr::select(
      new_track_id_i,
      new_track_id_pos,
      dur,
      last_heightdiff,
      mean_heightdiff,
      max_heightdiff,
      timediff,
      distance
    ) %>%
    set_names(
      "Select ID",
      "Link ID",
      "Duration",
      "Last HD",
      "Sit HD",
      "Stand HD",
      "Timediff",
      "Distance"
    ) %>%
    mutate_all(as.character)

  n_pos_ids <- nrow(df_feat)

  if (n_pos_ids > 0) {
    for (i in 1:n_pos_ids) {
      col <- scales::hue_pal()(n_pos_ids)[i]
      for (k in 2:8) {
        df_feat[i, k] <- paste0(
          '<span style="color:', col, '">', df_feat[i, k], "</span>"
        )
      }
    }
  }

  return(df_feat)
}

plot_sequence <- function(values) {
  if (is.null(values$dat)) {
    return(ggplot() +
      theme_classic())
  }
  track_seq <- values$dat %>%
    filter(category == "sure" | new_track_id == values$select_id) %>%
    group_by(new_track_id) %>%
    mutate(period = cumsum(
      in_check_tb != lag(in_check_tb, default = first(in_check_tb))
    )) %>%
    ungroup() %>%
    filter(in_check_tb) %>%
    group_by(new_track_id, period) %>%
    summarise(
      min_time = min(time),
      max_time = max(time)
    ) %>%
    ungroup() %>%
    mutate(
      across(
        c(min_time, max_time),
        ~ as.POSIXct(.x / 1000, origin = "1970-01-01")
      ),
      mark = ifelse(new_track_id == values$select_id, "id", "other")
    ) %>%
    arrange(min_time) %>%
    mutate(new_track_id = factor(
      new_track_id,
      levels = unique(new_track_id)
    )) %>%
    ungroup()
  track_seq_pl <- ggplot() +
    geom_segment(
      data = track_seq,
      mapping = aes(
        x = min_time,
        xend = max_time,
        y = new_track_id,
        group = period,
        color = mark
      ),
      linewidth = 2
    ) +
    geom_text(
      data = track_seq %>%
        group_by(new_track_id) %>%
        slice(1),
      mapping = aes(
        x = min_time,
        y = new_track_id,
        label = new_track_id
      ),
      size = 8 / cm(1),
      nudge_x = -10 * 60,
      nudge_y = 0
    )

  if (as.Date(values$clin$date[1]) > as.Date("2024-07-20")) {
    track_seq_pl <- track_seq_pl +
      geom_rect(
        data = values$clin,
        mapping = aes(
          xmin = start_time,
          xmax = completion_time,
          ymin = -Inf,
          ymax = Inf
        ),
        fill = "yellow",
        alpha = 0.2
      )
  }
  track_seq_pl <- track_seq_pl +
    theme_classic() +
    theme(
      axis.title.x = element_blank(),
      axis.title.y = element_blank(),
      axis.text.y = element_blank(),
      legend.position = "none"
    )
  return(track_seq_pl)
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
          "Possible" = 2,
          "Likely" = 3,
          "Sure" = 4,
          "Sputum"
        ),
        selected = 4,
        inline = TRUE
      ),
      actionButton("prev_id", "Previous ID"),
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
      div(
        tags$span("Time:", class = "bold-prefix"),
        textOutput("id_time"),
        style = "display: flex;"
      ),
      br(),
      div(
        tags$span("Height:", class = "bold-prefix"),
        textOutput("id_height"),
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
        choices = c(
          "not TB",
          "maybe",
          "possible",
          "likely",
          "sure",
          "sputum only"
        )
      ),
      actionButton("finish", "Enter"),
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
      tableOutput("links_table"),
      br(),
      plotOutput("sequence", inline = TRUE)
    )
  )
)



#### Shiny Sever ####
server <- function(input, output, session) {
  #### Load data ####
  values <- reactiveValues(
    dat = NULL, # tracking data
    clin = NULL, # clinical data
    ids_pu = NULL, # maybe
    ids_pc = NULL, # possible
    ids_du = NULL, # likely
    ids_dc = NULL, # sure
    select_id = NULL,
    prev_id = NULL,
    next_id = NULL,
    dat_i = NULL, # data of selected id
    dat_o = NULL, # data of other possible id links
    dat_o_feat = NULL, # feature data of other
    dat_os_feat = NULL, # subset of feature data
    dat_a = NULL # data of alternatives for selected other possible id link
  )

  observeEvent(input$file, {
    req(input$file)
    # get file
    file <- input$file

    # read tracking data
    values$dat <- read_tracking(file$datapath)

    # date
    file_date <- as.Date(as.POSIXct(
      values$dat$time[1] / 1000,
      origin = "1970-01-01"
    ))

    # update ID selection
    update_ids(values)
    values$select_id <- values$ids_dc[1]
    update_id_selection(session, values)

    # clinic ID count
    clin_sub <- subset(clinic, date == file_date)
    values$clin <- clin_sub
    values$clin$start_time <- as.POSIXct(values$clin$start_time)
    values$clin$completion_time <- as.POSIXct(values$clin$completion_time)
    n_clin_id <- n_distinct(values$clin$clinic_id)
    n_sputum <- n_clin_id - sum(values$clin$tb_test_res == "")
    min_time <- format(min(values$clin$start_time), "%H:%M")
    max_time <- format(max(values$clin$start_time), "%H:%M")
    output$clinical_ids <- renderText({
      paste0(
        n_sputum, "/", n_clin_id,
        " sputum [", min_time, " to ", max_time, "]"
      )
    })

    # directory to save file
    values$save_file <- paste0(
      "../data-clean/tracking/linked-tb/", file_date, ".csv"
    )
    output$saveto <- renderText({
      values$save_file
    })
  })

  #### Show subset of IDs ####
  observeEvent(input$show_ids, {
    show_type <- input$show_ids
    if (show_type == 1) {
      values$select_id <- values$ids_pu[1]
    } else if (show_type == 2) {
      values$select_id <- values$ids_pc[1]
    } else if (show_type == 3) {
      values$select_id <- values$ids_du[1]
    } else if (show_type == 4) {
      values$select_id <- values$ids_dc[1]
    } else {
      values$select_id <- values$ids_sp[1]
    }
    if (!is.null(values$select_id)) {
      update_id_selection(session, values)
    }
  })

  #### Select next ID ####
  # 1. Mark the selected ID as seen
  # 2. Update linkage
  # 3. Select next ID of subset
  # 4. Update selection
  observeEvent(input$next_id, {
    values$select_id <- values$next_id
    update_id_selection(session, values)
  })

  #### Select previous ID ####
  # 1. Select next ID of subset
  # 2. Update selection
  observeEvent(input$prev_id, {
    values$select_id <- values$prev_id
    update_id_selection(session, values)
  })

  #### ID Data ####
  # 1. Get selected ID and update info
  # 2. Silently update inputs without triggering new selection
  # 3. Compute features of possible links
  observeEvent(input$id, {
    # check if data and ID exist
    if (is.null(values$dat)) {
      return()
    } else {
      if (is.na(input$id)) {
        return()
      } else {
        values$select_id <- as.integer(input$id)
      }
    }

    # update ID data and information
    values$dat_i <- values$dat[values$dat$new_track_id == values$select_id, ]
    output$id_info <- renderText({
      update_info(values)
    })
    output$id_time <- renderText({
      update_time(values)
    })
    output$id_height <- renderText({
      update_height(values, input$direction)
    })
    output$date <- renderText({
      update_datetime(values)
    })
    output$tracking_ids <- renderText({
      update_counts(values)
    })

    # update several inputs
    updateRadioButtons(session, "direction", selected = 1)
    updateSliderTextInput(session, "time", selected = default_time)
    updateSliderTextInput(session, "distance", selected = default_dist)

    # compute features for others
    values$dat_o_feat <- filter_tracks(
      values$dat,
      values$select_id,
      input$direction
    )
    values$dat_o_feat <- compute_features(
      values$dat_o_feat,
      input$direction
    )
  })

  #### Switch linking direction ####
  observeEvent(input$direction, {
    if (
      !is.null(values$dat) &
        !is.null(values$dat_i) &
        !is.null(values$dat_o_feat)
    ) {
      # compute features for others
      values$dat_o_feat <- filter_tracks(
        values$dat,
        values$select_id,
        input$direction
      )
      values$dat_o_feat <- compute_features(
        values$dat_o_feat,
        input$direction
      )
    }
  })

  #### Show possible links ####
  observe({
    # get possible links
    if (!is.null(values$dat_o_feat)) {
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
    } else {
      updateSelectizeInput(session, inputId = "pos_id", choices = -1)
    }
  })

  #### Show possible alternatives ####
  observeEvent(input$pos_id, {
    if (!is.null(values$dat_os_feat)) {
      pos_id <- as.integer(input$pos_id)
      pos_id_td <- values$dat_os_feat[
        values$dat_os_feat$new_track_id_other == pos_id,
        "timediff"
      ]
      pos_id_dist <- values$dat_os_feat[
        values$dat_os_feat$new_track_id_other == pos_id,
        "distance"
      ]
      values$dat_a_feat <- filter_tracks(
        values$dat,
        pos_id,
        ifelse(input$direction == 1, 2, 1)
      ) %>%
        filter(new_track_id_other != values$select_id)
      if (nrow(values$dat_a_feat) > 0 & length(pos_id_td) > 0) {
        values$dat_a_feat <- compute_features(
          values$dat_a_feat,
          ifelse(input$direction == 1, 2, 1)
        )
        dat_as_feat <- values$dat_a_feat %>%
          filter(
            timediff <= pmax(pos_id_td, min_alt_time),
            distance <= pmax(pos_id_dist, min_alt_dist)
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
  # 1. Link IDs and choose the selected ID as the new ID for both
  # 2. Keep seen and TB status
  observeEvent(input$link, {
    pos_id <- as.integer(input$pos_id)
    if (pos_id == -1) {
      shinyalert(
        "Error: -1",
        "No ID selected to link with.",
        type = "error",
        timer = 1000
      )
    } else {
      shinyalert(
        "Success",
        paste("Linking ID", values$select_id, "to", pos_id, "."),
        type = "success",
        timer = 1000
      )
      link_id_track <- values$dat$new_track_id %in% c(values$select_id, pos_id)
      select_id_track <- values$dat$new_track_id == values$select_id
      values$dat$new_track_id[link_id_track] <- values$select_id
      values$dat$category[link_id_track] <- values$dat$category[select_id_track][1]
      update_linkage(values)
      update_ids(values)
      update_id_selection(session, values)
    }
  })

  #### Unlink last ID ####
  # 1. Unlink last ID and choose second last as new ID
  # 2. Mark the unlinked ID as seen and move to maybe TB
  # 3. Do not update seen of the selected ID, neither TB status
  observeEvent(input$unlink_last, {
    link_id_track <- values$dat$new_track_id == values$select_id
    linked_ids <- unique(values$dat$track_id[link_id_track])
    last_id <- linked_ids[length(linked_ids)]
    values$select_id <- linked_ids[length(linked_ids) - 1]
    shinyalert(
      "Success",
      paste("Unlinking ID", last_id, "from", values$select_id),
      type = "info",
      timer = 1000
    )
    unlink_id_track <- values$dat$track_id %in% setdiff(linked_ids, last_id)
    last_id_track <- values$dat$track_id == last_id
    values$dat$new_track_id[unlink_id_track] <- values$select_id
    values$dat$new_track_id[last_id_track] <- last_id
    values$dat$category[last_id_track] <- "possible"
    update_linkage(values)
    update_ids(values)
    update_id_selection(session, values)
  })

  #### Unlink first ID ####
  # 1. Unlink first ID and choose last as new ID
  # 2. Mark the unlinked ID as seen and move to maybe TB
  # 3. Do not update seen of the selected ID, neither TB status
  observeEvent(input$unlink_first, {
    link_id_track <- values$dat$new_track_id == values$select_id
    linked_ids <- unique(values$dat$track_id[link_id_track])
    first_id <- linked_ids[1]
    values$select_id <- linked_ids[length(linked_ids)]
    shinyalert(
      "Success",
      paste("Unlinking ID", first_id, "from", values$select_id),
      type = "info",
      timer = 1000
    )
    unlink_id_track <- values$dat$track_id %in% setdiff(linked_ids, first_id)
    first_id_track <- values$dat$track_id == first_id
    values$dat$new_track_id[unlink_id_track] <- values$select_id
    values$dat$new_track_id[first_id_track] <- first_id
    values$dat$category[first_id_track] <- "possible"
    update_linkage(values)
    update_ids(values)
    update_id_selection(session, values)
  })


  #### Categorize track  ####
  observeEvent(input$finish, {
    shinyalert(
      "Success",
      paste("ID", values$select_id, "is", input$is_tb),
      type = "info",
      timer = 1000
    )
    values$dat$category[values$dat$new_track_id == values$select_id] <- input$is_tb
    update_linkage(values)
    update_ids(values)
    values$select_id <- values$next_id
    update_id_selection(session, values)
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
    height = 600,
    width = 600 * 3.33
  )

  #### Table ####
  output$links_table <- renderTable(
    {
      table_ids(values$dat_i, values$dat_o, input$direction)
    },
    sanitize.text.function = function(x) x
  )

  output$sequence <- renderPlot(
    {
      plot_sequence(values)
    },
    height = 400,
    width = 1000
  )
}





#### Shiny App ####
shinyApp(ui = ui, server = server)
