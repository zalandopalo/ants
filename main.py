import cv2
import math as m
import random
import sympy.geometry
import numpy as np
from sympy import Point
import pickle


def image_preparation(path_to_img):
    path_to_img = cv2.imread(path_to_img)
    path_to_img = cv2.medianBlur(path_to_img, 7)
    # prepare img
    img_gray = cv2.cvtColor(path_to_img, cv2.COLOR_BGR2GRAY)
    thresh = 100

    ret, img_thresh = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find minimum and maximum size contour
    min_val = m.inf
    max_val = 0
    for contour in contours:
        if contour.shape[0] < min_val:
            min_val = contour.shape[0]
        if contour.shape[0] > max_val:
            max_val = contour.shape[0]

    # make nods
    approx_contours = []
    for contour in contours:
        esp = 0.02 / (contour.shape[0] / min_val)
        arclen = cv2.arcLength(contour, True)
        epsilon = arclen * esp
        approx_contours.append(cv2.approxPolyDP(contour, epsilon, True))

    return approx_contours


def min_max_contour_points(figs_nods_cords):
    import math
    x_max = - math.inf
    x_min = math.inf
    y_max = - math.inf
    y_min = math.inf
    try:
        for figure in figs_nods_cords:
            for elem in figure:
                x0 = elem[0][0]
                y0 = elem[0][1]
                if x0 > x_max:
                    x_max = x0
                if x0 < x_min:
                    x_min = x0
                if y0 > y_max:
                    y_max = y0
                if y0 < y_min:
                    y_min = y0
    except:
        for figure in figs_nods_cords:
            x0 = figure[0]
            y0 = figure[1]
            if x0 > x_max:
                x_max = x0
            if x0 < x_min:
                x_min = x0
            if y0 > y_max:
                y_max = y0
            if y0 < y_min:
                y_min = y0
    return [x_max, y_max, x_min, y_min]


def line_outfig(line_from_to: sympy.geometry.Segment, lines_in_figs: list, x_max_from_figs):
    """
    for segment check ( in or out)
    :param line_from_to: sympy.geometry.Segment
    :param lines_in_figs: collided contours
    :param x_max_from_figs:  for left fast check - inside fig or not
    :return: True if outside
    """

    from sympy.geometry import Point, Segment

    mid_point = line_from_to.midpoint
    y_mid_point = mid_point.y
    l_control = 0
    control_point = Point(x_max_from_figs, y_mid_point)
    control_seg = Segment(mid_point, control_point)

    try:
        for fig in lines_in_figs:
            for line in fig:
                control = control_seg.intersection(line)
                if control != []:
                    for c in control:
                        if type(c) is type(control_point):
                            l_control += 1
                else:
                    continue
    except:
        for line in lines_in_figs:
            control = control_seg.intersection(line)
            if control != []:
                for c in control:
                    if type(c) is type(control_point):
                        l_control += 1
            else:
                continue

    if l_control // 2 == 0 or l_control == 0:
        return True
    else:
        return False


def make_lines_from_cv_cont(figures_cont):
    from sympy.geometry import Point, Segment
    symp_cont = []
    for figure in figures_cont:
        symp_fig = []
        for counter in range(-1, len(figure)):
            try:
                p1 = Point(figure[counter][0][0], figure[counter][0][1])
                p2 = Point(figure[counter + 1][0][0], figure[counter + 1][0][1])
                s = Segment(p1, p2)
                symp_fig.append(s)
                sr = Segment(p2, p1)
                symp_fig.append(sr)
                # make lines for interception check:
            except:
                p1 = Point(figure[counter][0][0], figure[counter][0][1])
                p2 = Point(figure[0][0][0], figure[0][0][1])
                s = Segment(p1, p2)
                symp_fig.append(s)
                sr = Segment(p2, p1)
                symp_fig.append(sr)
        symp_cont.append(symp_fig)
    return symp_cont


def make_points(countours):
    points = []
    for figure in countours:
        for point in figure:
            points.append(point[0])
    return points


def build_graph_for_current_position(points_in_all_contours, contour_lines, current_pose, bot_color):
    from sympy.geometry import Point, Segment
    pts = [Point(point[0], point[1]) for point in points_in_all_contours]
    all_lines = []
    for fig in contour_lines:
        for line in fig:
            all_lines.append(line)
    match type(current_pose):
        case sympy.geometry.Point2D:
            point_from = current_pose
        case list():
            ptx = current_pose[0]
            pty = current_pose[1]
            point_from = Point(ptx, pty)
        case _:
            print("KRYA")

    avalible_moves = []
    lenths = []

    for j in range(len(pts)):
        if point_from == pts[j]:
            continue
        point_to = pts[j]
        seg_from_to = Segment(point_from, point_to)
        if seg_from_to in all_lines:
            avalible_moves.append(point_to)
            lenths.append(seg_from_to.length)
            continue
        try:
            figs_in_time = []
            point_to_time = point_from
            inters = []
            for fig in contour_lines:
                try:
                    for line in fig:
                        try:
                            intersection = seg_from_to.intersection(line)[0]
                        except:
                            intersection = seg_from_to.intersection(line)
                        match type(intersection):
                            case sympy.geometry.Segment2D:
                                point_to_time = intersection.points[1]
                                if point_to_time not in inters:
                                    inters.append(point_to_time)
                                if fig not in figs_in_time:
                                    figs_in_time.append(fig)
                                continue

                            case sympy.geometry.Point2D:
                                if intersection not in pts:
                                    raise StopIteration

                                elif (intersection in pts) and (intersection != point_from):
                                    if line_outfig(Segment(point_to_time, intersection), fig, minmax[0]):
                                        if intersection not in inters:
                                            inters.append(intersection)
                                        if fig not in figs_in_time:
                                            figs_in_time.append(fig)
                                        continue
                                    else:
                                        raise StopIteration

                                elif intersection == point_from:

                                    continue
                            case _:
                                continue
                    continue
                except StopIteration:
                    raise StopIteration
            # проверку на внутренность линии

            avalible_moves.append(point_to)
            pfx, pfy = point_from.x, point_from.y
            ptx1, pty1 = point_to.x, point_to.y
            leng = round(m.sqrt(((ptx1 - pfx) ** 2) + ((pty1 - pfy) ** 2)), 3)
            lenths.append(leng)
        except StopIteration:
            continue
    return avalible_moves, lenths


def add_target_point(points, target):
    points.append(np.array([int(target[0]), int(target[1])]))
    return points


class Agent:

    def __init__(self, x_pose, y_pose):
        self.m_type = None
        self.w = None
        self.h = None

        self.x_pose = x_pose
        self.y_pose = y_pose
        self.current_point = Point(self.x_pose, self.y_pose)
        self.first_pose = Point(x_pose, y_pose)
        self.prev_pose = Point(x_pose, y_pose)

        self.pher_line = 1
        self.way = [[x_pose, y_pose]]
        self.color = None
        self.alpha = 0.5
        self.betta = 0.5
        self.target = Point(600, 150)
        self.phers = dict()

    def set_pher_line(self, lvl):
        """
        Set pheromones lvl
        :param lvl: float
        :return: None
        """
        self.pher_line = lvl

    def set_target(self, point):
        """
        set Target for bot
        :param point: list [x,y] or sympy.Geometry.Point(x,y)
        :return:
        """
        match type(point):
            case list():
                ptx = point[0]
                pty = point[1]
                self.target = Point(ptx, pty)
            case sympy.geometry.Point2D():
                self.target = point

        return self.target

    def set_color(self, color: tuple):
        """
        set bot color
        :param color: (B,G,R)
        :return: None
        """

        self.color = color

    def set_size(self, h: int = 1, w: int = 1):
        """
        :param h: default = 1
        :param w: default = 1
        :return: add size to bot
        """
        if h != 1:
            self.h = h
        else:
            self.h = 1
        if w != 1:
            self.w = w
        else:
            self.w = 1

    def ret_way(self):
        return self.way

    def ret_phers(self):
        return self.phers

    def next_iteration(self):
        """
        Reset bot variables
        :return:
        """
        self.x_pose, self.y_pose = self.first_pose.x, self.first_pose.y
        self.current_point = self.first_pose
        self.way.clear()
        self.phers.clear()

    def move_with_ants(self, available_moves, lengths, phers_for_ways: dict):
        """
        chose next direction, with ant algorithm
        :param available_moves: list [Point....]
        :param lengths: list [float ....]
        :param phers_for_ways: f"available_moves[j]" : float(lvl)
        :return: list(list(from_x, from_y), list(to_x, to_y))
        """
        aval_m = []
        Pij = []
        sums = 0
        for j in range(len(available_moves)):
            if [int(available_moves[j].x), int(available_moves[j].y)] not in self.way:
                aval_m.append(available_moves[j])
                sums += ((float(phers_for_ways.get(f"{available_moves[j]}"))) ** self.alpha) * (
                        float((lengths.get(f"{self.current_point}")[j])) ** self.betta)
            else:
                continue
        point_from_x, point_from_y = self.x_pose, self.y_pose

        for i in range(len(aval_m)):
            if [int(aval_m[i].x), int(aval_m[i].y)] not in self.way:
                Pnn = (((float(phers_for_ways.get(f"{aval_m[i]}"))) ** self.alpha) * (
                        float((lengths.get(f"{self.current_point}")[i])) ** self.betta)) / sums
                Pij.append(Pnn)

        try:
            selected_move = random.choices(aval_m, Pij)
            self.prev_pose = self.current_point
            self.current_point = selected_move[0]
            self.phers.update({f"{selected_move[0]}": float(
                phers_for_ways.get(f"{selected_move[0]}")) + self.pher_line})

            self.x_pose = int(selected_move[0].x)
            self.y_pose = int(selected_move[0].y)

            self.way.append([self.x_pose, self.y_pose])
            return [[point_from_x, point_from_y], [self.x_pose, self.y_pose]]

        except:
            de = [int(self.current_point.x), int(self.current_point.y)]
            self.way.remove(de)
            self.current_point = self.prev_pose
            self.prev_pose = Point(self.way[-1][0], self.way[-1][1])
            self.x_pose = self.way[-1][0]
            self.y_pose = self.way[-1][1]
            return [[point_from_x, point_from_y], [self.x_pose, self.y_pose]]


# open / save data for faster work
try:
    p = open('phers_for_moves', 'rb')
    phers_for_moves = pickle.load(p)
    p.close()
except:
    phers_for_moves = dict()
try:
    v = open('vis_graph', 'rb')
    ls = open('lenths_graph', 'rb')
    vis_graph = pickle.load(v)
    lenths_graph = pickle.load(ls)
    v.close()
    ls.close()
except:
    vis_graph = dict()
    lenths_graph = dict()

target_point = Point(600, 150)

phers_for_moves.update({f"{target_point}": 10})

img_path = "Picture1.png"
img = cv2.imread("Picture1.png")
approx_img = image_preparation(img_path)
lines_from_cont = make_lines_from_cv_cont(approx_img)
points = make_points(approx_img)
minmax = min_max_contour_points(approx_img)

cv2.drawContours(img, approx_img, -1, color=(13, 13, 115), thickness=2)
# for fig in approx_img:
#   for pt in fig:
#       cv2.circle(img, (pt[0][0], pt[0][1]), 3, (13, 115, 13), -1)
# setting bots
iterations = 10

points = add_target_point(points, [600, 150])
cv2.circle(img, [600, 150], 5, (250, 250, 250), 3)

img_curent = img.copy()

agent1 = Agent(50, 350)
agent1.set_target([600, 150])
agent1.set_color((133, 21, 199))

agent2 = Agent(50, 350)
agent2.set_color((0, 69, 255))
agent1.set_target([600, 150])

agent3 = Agent(50, 350)
agent3.set_target([600, 150])
agent3.set_color((255, 255, 0))

cv2.circle(img, [agent1.x_pose, agent1.y_pose], 5, agent1.color, 3)
cv2.circle(img, [agent2.x_pose, agent2.y_pose], 5, agent2.color, 3)
cv2.circle(img, [agent3.x_pose, agent3.y_pose], 5, agent3.color, 3)


def get_vals(agent, nodes, lines_from_cont):
    if f"{agent.current_point}" not in nodes.keys():
        avalible_nods, lenghts_to_nods = build_graph_for_current_position(points, lines_from_cont, agent.current_point,
                                                                          agent.color)
        vis_graph.update({f"{agent1.current_point}": avalible_nods})
        lenths_graph.update({f"{agent1.current_point}": lenghts_to_nods})
    else:
        avalible_nods, lenghts_to_nods = vis_graph.get(f"{agent.current_point}"), lenths_graph.get(
            f"{agent.current_point}")

    for i in range(len(avalible_nods)):
        if phers_for_moves.get(f"{avalible_nods[i]}") is None:
            phers_for_moves.update({f"{avalible_nods[i]}": 0.1})


shortest_len = m.inf
shortest_way = []


it = 0
while True:
    #img_curent_timed = cv2.imread("fastest.png")
    img_curent_timed = img.copy()
    fastest_way = img.copy()
    move_img = img.copy()
    if shortest_len != m.inf:
        x0, y0 = shortest_way[0]
        for i in range(1, len(shortest_way)):
            cv2.line(img_curent_timed, [x0, y0], shortest_way[i], (1, 150, 1), 1)
            x0, y0 = shortest_way[i]

    # // graphs

    if f"{agent1.current_point}" not in vis_graph.keys():
        aval1, lenths1 = build_graph_for_current_position(points, lines_from_cont, agent1.current_point, agent1.color)
        vis_graph.update({f"{agent1.current_point}": aval1})
        lenths_graph.update({f"{agent1.current_point}": lenths1})

    else:
        aval1, lenths1 = vis_graph.get(f"{agent1.current_point}"), lenths_graph.get(f"{agent1.current_point}")
    for i in range(len(aval1)):
        if phers_for_moves.get(f"{aval1[i]}") is None:
            phers_for_moves.update({f"{aval1[i]}": 1})

    if f"{agent2.current_point}" not in vis_graph.keys():
        aval2, lenths2 = build_graph_for_current_position(points, lines_from_cont, agent2.current_point, agent2.color)
        vis_graph.update({f"{agent2.current_point}": aval2})
        lenths_graph.update({f"{agent2.current_point}": lenths2})

    else:
        aval2, lenths2 = vis_graph.get(f"{agent2.current_point}"), lenths_graph.get(f"{agent2.current_point}")

    for i in range(len(aval2)):
        if phers_for_moves.get(f"{aval2[i]}") is None:
            phers_for_moves.update({f"{aval2[i]}": 1})

    if f"{agent3.current_point}" not in vis_graph.keys():
        aval3, lenths3 = build_graph_for_current_position(points, lines_from_cont, agent3.current_point, agent3.color)
        vis_graph.update({f"{agent3.current_point}": aval3})
        lenths_graph.update({f"{agent3.current_point}": lenths3})
    else:
        aval3, lenths3 = vis_graph.get(f"{agent3.current_point}"), lenths_graph.get(f"{agent3.current_point}")
    for i in range(len(aval3)):
        if phers_for_moves.get(f"{aval3[i]}") is None:
            phers_for_moves.update({f"{aval3[i]}": 1})

    for i in range(len(aval1)):
        cv2.line(move_img, [int(agent1.current_point.x), int(agent1.current_point.y)],
                 [int(aval1[i].x), int(aval1[i].y)], agent1.color, 1)
    for i in range(len(aval2)):
        cv2.line(move_img, [int(agent2.current_point.x), int(agent2.current_point.y)],
                 [int(aval2[i].x), int(aval2[i].y)], agent2.color, 1)
    for i in range(len(aval3)):
        cv2.line(move_img, [int(agent3.current_point.x), int(agent3.current_point.y)],
                 [int(aval3[i].x), int(aval3[i].y)], agent3.color, 1)

    line_move1 = agent1.move_with_ants(aval1, lenths_graph, phers_for_moves)
    line_move2 = agent2.move_with_ants(aval2, lenths_graph, phers_for_moves)
    line_move3 = agent3.move_with_ants(aval3, lenths_graph, phers_for_moves)
    # cv2.imshow("movs", move_img)
    # cv2.waitKey()

    # \\ graphs end

    # // current position
    cv2.circle(img_curent_timed, [line_move1[1][0], line_move1[1][1]], 6, agent1.color, 3)
    cv2.putText(img_curent_timed, "bot 1", [line_move1[1][0] + 10, line_move1[1][1] + 10], cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 0, 0), 1)
    cv2.circle(img_curent_timed, [line_move2[1][0], line_move2[1][1]], 5, agent2.color, 3)
    cv2.putText(img_curent_timed, "bot 2", [line_move2[1][0] + 10, line_move2[1][1]], cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 0, 0), 1)
    cv2.circle(img_curent_timed, [line_move3[1][0], line_move3[1][1]], 4, agent3.color, 3)
    cv2.putText(img_curent_timed, "bot 3", [line_move3[1][0] + 10, line_move3[1][1] - 10], cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 0, 0), 1)
    #cv2.imshow("timed", img_curent_timed)
    cv2.imwrite(f"timed{it}.png", img_curent_timed)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    it += 1
    # \\ current position

    # open / save data for faster work
    p = open('phers_for_moves', 'wb')
    v = open('vis_graph', 'wb')
    ls = open('lenths_graph', 'wb')
    pickle.dump(phers_for_moves, p)
    pickle.dump(vis_graph, v)
    pickle.dump(lenths_graph, ls)
    p.close()
    v.close()
    ls.close()

    # // detect route end
    if agent1.current_point == agent1.target:
        ag1 = agent1.ret_way()
        ph = agent1.ret_phers()

        if len(ag1) <= shortest_len:
            shortest_len = len(ag1)
            shortest_way = ag1.copy()
            shortest_way.insert(0, [int(agent1.first_pose.x), int(agent1.first_pose.y)])

        for key in ph.keys():
            phers_for_moves.update({f"{key}": phers_for_moves.get(f"{key}") + ph.get(f"{key}")})

        x0, y0 = int(agent1.first_pose.x), int(agent1.first_pose.y)

        for w in range(0, len(ag1)):
            cv2.circle(fastest_way, ag1[w], 3, agent1.color, 3)
            cv2.line(fastest_way, [x0, y0], ag1[w], agent1.color, 2)
            x0, y0 = ag1[w]

        for key in phers_for_moves.keys():
            if not m.isclose(float(phers_for_moves.get(key)), 1, abs_tol=0.01):
                phers_for_moves.update({f"{key}": float(phers_for_moves.get(key)) - 0.1})
        cv2.imshow("fastest", fastest_way)
        cv2.imwrite(f"fastest{it}{it}.png", fastest_way)


        # cv2.waitKey()

        agent1.next_iteration()

    if agent2.current_point == agent2.target:
        ag2 = agent2.ret_way()
        ph = agent2.ret_phers()

        if len(ag2) <= shortest_len:
            shortest_len = len(ag2)
            shortest_way = ag2.copy()
            shortest_way.insert(0, [int(agent2.first_pose.x), int(agent3.first_pose.y)])

        for key in ph.keys():
            phers_for_moves.update({f"{key}": phers_for_moves.get(f"{key}") + ph.get(f"{key}")})
        x0, y0 = int(agent2.first_pose.x), int(agent2.first_pose.y)

        for w in range(0, len(ag2)):
            cv2.circle(fastest_way, ag2[w], 3, agent2.color, 3)
            cv2.line(fastest_way, [x0, y0], ag2[w], agent2.color, 2)
            x0, y0 = ag2[w]

        for key in phers_for_moves.keys():
            if not m.isclose(float(phers_for_moves.get(key)), 1, abs_tol=0.01):
                phers_for_moves.update({f"{key}": float(phers_for_moves.get(key)) - 0.1})
        cv2.imshow("fastest", fastest_way)
        cv2.imwrite(f"fastest{it}{it}.png", fastest_way)
        # cv2.waitKey()

        agent2.next_iteration()

    if agent3.current_point == agent3.target:
        ag3 = agent3.ret_way()
        ph = agent3.ret_phers()
        if len(ag3) <= shortest_len:
            shortest_len = len(ag3)
            shortest_way = ag3.copy()
            shortest_way.insert(0, [int(agent3.first_pose.x), int(agent3.first_pose.y)])

        for key in ph.keys():
            phers_for_moves.update({f"{key}": phers_for_moves.get(f"{key}") + ph.get(f"{key}")})
        x0, y0 = int(agent3.first_pose.x), int(agent3.first_pose.y)

        for w in range(0, len(ag3)):
            cv2.circle(fastest_way, ag3[w], 3, agent3.color, 3)
            cv2.line(fastest_way, [x0, y0], ag3[w], agent3.color, 2)
            x0, y0 = ag3[w]
        for key in phers_for_moves.keys():
            if not m.isclose(float(phers_for_moves.get(key)), 1, abs_tol=0.1):
                phers_for_moves.update({f"{key}": float(phers_for_moves.get(key)) - 0.2})
        agent3.next_iteration()
        cv2.imshow("fastest", fastest_way)
        cv2.imwrite(f"fastest{it}{it}.png", fastest_way)

        # cv2.waitKey()

    # \\ detect route end

    cv2.waitKey()
    cv2.destroyAllWindows()
