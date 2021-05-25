"""
Copyright 2021, Eyuell H Gebremedhin

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

def get_header(vid_name, vid_num, fr_size):
    x = str()
    x = x + '<?xml version="1.0" encoding="UTF-8"?> <annotations> <version>1.1</version> <meta> <task>'
    x = x + ' <name>' + vid_num + '</name> <size>' + str(fr_size) + '</size> <labels>'
    x = x + ' <label> <name>sign</name> <attributes> <attribute>@select=type:ped_blue,ped_yellow,ped_white,'
    x = x + 'ped_text,stop_sign,bus_stop,train_stop,construction,other</attribute> <attribute>@text=id:</attribute>'
    x = x + '</attributes> </label> <label> <name>vehicle</name> <attributes> <attribute>@select=type:car,truck,bus,'
    x = x + 'train,bicycle,bike</attribute> <attribute>@text=id:</attribute></attributes> </label> <label>'
    x = x + '<name>crosswalk</name> <attributes><attribute>@text=id:</attribute> </attributes> </label>'
    x = x + '<label> <name>transit_station</name> <attributes> <attribute>@text=id:</attribute> </attributes> </label>'
    x = x + '<label> <name>traffic_light</name> <attributes> <attribute>@select=type:regular,transit,pedestrian</attribute>'
    x = x + '<attribute>@text=id:</attribute> <attribute>~select=state:__undefined__,green,yellow,red</attribute>'
    x = x + '</attributes> </label> <label> <name>pedestrian</name> <attributes> <attribute>~select=cross:not-crossing,'
    x = x + 'crossing,crossing-irrelevant</attribute> <attribute>~select=occlusion:none,part,full</attribute>'
    x = x + '<attribute>@text=id:</attribute> <attribute>~select=action:standing,walking</attribute>'
    x = x + '<attribute>~select=gesture:__undefined__,hand_ack,hand_yield,hand_rightofway,nod,other</attribute>'
    x = x + '<attribute>~select=look:not-looking,looking</attribute> </attributes> </label> </labels>'
    x = x + '<owner><username>Eyuell</username> <email>gusgebey@student.gu.se</email></owner> <original_size>'
    x = x + '<width>1920</width> <height>1280</height> </original_size> </task> </meta> <track label="pedestrian">'

    return x