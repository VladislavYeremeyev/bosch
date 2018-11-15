class Eq:
	cnt = 0
	eq = []

	def openFile(self, fullPath): #открывает файл
		return open(fullPath, 'r')

	def __init__(self, fullPath): #инициализация класса эквивалентности, запись данных из файла
		f = self.openFile(fullPath)
		data = f.readlines()
		self.cnt = len(data)
		self.eq = [0]*self.cnt
		for i in range(len(data)):
			self.eq[i] = []*len(data[i].split(','))
			self.eq[i] = data[i].split(',')
		f.close()

	def idEq(self, a, b): #строгая эквивалентность, проверка по типу объекта
		return a == b

	def levelEq(self, a, b): #уровень эквивалентности от 1 до 9 (1 - ничего, 9 - по всем параметрам)
		i = 1
		for obj in self.eq:
			if self.Ex(a, obj) & self.Ex(b, obj):
				i = i + 1
		return i

	def Ex(self, a, array): #проверка есть ли элемент а в массиве
		for obj in array:
			if obj == a:
				return True
		return False

	def trueEq(self, a, b): #возвращает уровень эквивалентности 0 - идеальный, 1-9
		if self.idEq(a, b):
			return 0
		else:
			return self.levelEq(a, b)

	def maxEqObj(self, a, b): #возвращает максимально эквивалентный объект для а (1е фото) из массива объектов b (2е фото)
		lvlEq = self.levelEq(a.id,b[0].id) #работает по id
		for objB in b:
			if(self.levelEq(a.id,objB.id) == 0)|(self.levelEq(a.id,objB.id)>lvlEq):
				if(self.levelEq(a.id,objB.id) == 0):
					return 0
				lvlEq = self.levelEq(a.id, objB.id)
				maxEqB = objB
		return maxEqB
